"""Battery physical simulation, energy balance invariants, and settlement."""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.utils import setup_logger, mw_to_mwh, calculate_cost

logger = setup_logger(__name__)


@dataclass
class BatteryConfig:
    """Battery energy storage system parameters."""
    energy_capacity_mwh: float = 10.0
    max_charge_power_mw: float = 2.5
    max_discharge_power_mw: float = 2.5
    round_trip_efficiency: float = 0.88
    soc_min: float = 0.10
    soc_max: float = 0.90
    soc_initial: float = 0.50
    soc_terminal: float = 0.50
    grid_import_limit_mw: float = 6.0
    grid_export_limit_mw: float = 5.0
    retail_adder_eur_mwh: float = 20.0
    export_ratio: float = 1.0
    degradation_cost_eur_mwh: float = 0.0
    allow_negative_pricing: bool = False

    @property
    def charge_efficiency(self) -> float:
        return float(np.sqrt(self.round_trip_efficiency))

    @property
    def discharge_efficiency(self) -> float:
        return float(np.sqrt(self.round_trip_efficiency))


@dataclass
class SimulationResult:
    """Encapsulates time series traces and summary metrics for a simulation run."""
    strategy_name: str
    timeline: pd.DataFrame
    total_cost_eur: float
    grid_import_cost_eur: float
    grid_export_revenue_eur: float
    degradation_cost_eur: float
    total_grid_import_mwh: float
    total_grid_export_mwh: float
    total_solar_generated_mwh: float
    total_load_mwh: float
    total_charged_mwh: float
    total_discharged_mwh: float
    total_curtailed_mwh: float
    equivalent_full_cycles: float
    round_trip_losses_mwh: float
    max_energy_balance_residual: float
    constraint_violations: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class BatterySimulator:
    """Simulates physical battery dispatch and grid settlement step-by-step."""

    def __init__(self, config: BatteryConfig):
        self.cfg = config
        self.reset()

    def reset(self, initial_soc: Optional[float] = None) -> None:
        """Resets battery state to initial condition."""
        if initial_soc is not None:
            self.current_soc = float(np.clip(initial_soc, self.cfg.soc_min, self.cfg.soc_max))
        else:
            self.current_soc = float(self.cfg.soc_initial)

    def step(
        self,
        solar_mw: float,
        load_mw: float,
        price_eur_mwh: float,
        requested_p_charge_mw: float,
        requested_p_discharge_mw: float,
        dt_hours: float = 1.0,
    ) -> Dict[str, float]:
        """Simulates a single time step, enforcing physical and grid constraints.

        Returns:
            Dictionary of realized powers, energy flows, SoC, and settlement cost.
        """
        # 1. Enforce no simultaneous charge and discharge
        if requested_p_charge_mw > 1e-4 and requested_p_discharge_mw > 1e-4:
            raise ValueError(
                f"Simultaneous charge ({requested_p_charge_mw:.3f} MW) and discharge "
                f"({requested_p_discharge_mw:.3f} MW) requested! Physical violation."
            )

        # 2. Enforce power ratings
        p_ch_cmd = float(np.clip(requested_p_charge_mw, 0.0, self.cfg.max_charge_power_mw))
        p_dis_cmd = float(np.clip(requested_p_discharge_mw, 0.0, self.cfg.max_discharge_power_mw))

        # 3. Enforce SoC boundaries
        max_energy_ch = (self.cfg.soc_max - self.current_soc) * self.cfg.energy_capacity_mwh
        max_p_ch_soc = max(0.0, max_energy_ch / (self.cfg.charge_efficiency * dt_hours))
        actual_p_ch = min(p_ch_cmd, max_p_ch_soc)

        max_energy_dis = (self.current_soc - self.cfg.soc_min) * self.cfg.energy_capacity_mwh
        max_p_dis_soc = max(0.0, (max_energy_dis * self.cfg.discharge_efficiency) / dt_hours)
        actual_p_dis = min(p_dis_cmd, max_p_dis_soc)

        # 4. Energy Management System (EMS) Grid Headroom Enforcer:
        # If charging would cause grid import to exceed grid limit, throttle charging
        net_load_mw = load_mw - solar_mw
        if net_load_mw > 0:
            grid_import_headroom = max(0.0, self.cfg.grid_import_limit_mw - net_load_mw)
            actual_p_ch = min(actual_p_ch, grid_import_headroom)

        # 5. Update SoC
        delta_soc = (
            (actual_p_ch * self.cfg.charge_efficiency * dt_hours)
            - (actual_p_dis / self.cfg.discharge_efficiency * dt_hours)
        ) / self.cfg.energy_capacity_mwh
        new_soc = float(np.clip(self.current_soc + delta_soc, self.cfg.soc_min, self.cfg.soc_max))
        self.current_soc = new_soc

        # 6. Grid interaction & settlement
        grid_req_mw = net_load_mw + actual_p_ch - actual_p_dis

        actual_p_imp = 0.0
        actual_p_exp = 0.0
        p_curtail = 0.0

        if grid_req_mw >= 0:
            actual_p_imp = min(grid_req_mw, self.cfg.grid_import_limit_mw)
        else:
            surplus_mw = -grid_req_mw
            if price_eur_mwh < 0 and not self.cfg.allow_negative_pricing:
                p_curtail = surplus_mw
                actual_p_exp = 0.0
            else:
                actual_p_exp = min(surplus_mw, self.cfg.grid_export_limit_mw)
                p_curtail = surplus_mw - actual_p_exp

        # 7. Physical energy balance residual
        sources = solar_mw + actual_p_dis + actual_p_imp
        sinks = load_mw + actual_p_ch + actual_p_exp + p_curtail
        residual = abs(sources - sinks)

        # 8. Financial settlement
        import_price = price_eur_mwh + self.cfg.retail_adder_eur_mwh
        export_price = price_eur_mwh * self.cfg.export_ratio

        e_imp_mwh = mw_to_mwh(actual_p_imp, dt_hours)
        e_exp_mwh = mw_to_mwh(actual_p_exp, dt_hours)
        e_ch_mwh = mw_to_mwh(actual_p_ch, dt_hours)
        e_dis_mwh = mw_to_mwh(actual_p_dis, dt_hours)

        import_cost_eur = e_imp_mwh * import_price
        export_revenue_eur = e_exp_mwh * export_price
        degradation_eur = (e_ch_mwh + e_dis_mwh) * self.cfg.degradation_cost_eur_mwh
        net_step_cost_eur = import_cost_eur - export_revenue_eur + degradation_eur

        return {
            "p_charge_mw": actual_p_ch,
            "p_discharge_mw": actual_p_dis,
            "p_import_mw": actual_p_imp,
            "p_export_mw": actual_p_exp,
            "p_curtail_mw": p_curtail,
            "soc": new_soc,
            "residual_mw": residual,
            "import_cost_eur": import_cost_eur,
            "export_revenue_eur": export_revenue_eur,
            "degradation_cost_eur": degradation_eur,
            "net_cost_eur": net_step_cost_eur,
        }

    def simulate(
        self,
        timeline_df: pd.DataFrame,
        strategy_name: str = "Simulation",
        p_charge_col: str = "p_charge_req",
        p_discharge_col: str = "p_discharge_req",
    ) -> SimulationResult:
        """Runs the simulation across an entire scheduled timeline DataFrame."""
        self.reset()
        records = []
        violations = 0
        max_residual = 0.0

        for ts, row in timeline_df.iterrows():
            solar = float(row["solar_mw"])
            load = float(row["load_mw"])
            price = float(row["price_eur_mwh"])
            p_ch_req = float(row.get(p_charge_col, 0.0))
            p_dis_req = float(row.get(p_discharge_col, 0.0))

            res = self.step(
                solar_mw=solar,
                load_mw=load,
                price_eur_mwh=price,
                requested_p_charge_mw=p_ch_req,
                requested_p_discharge_mw=p_dis_req,
                dt_hours=1.0,
            )

            if res["residual_mw"] > 1e-4:
                violations += 1
            max_residual = max(max_residual, res["residual_mw"])

            res["timestamp"] = ts
            res["solar_mw"] = solar
            res["load_mw"] = load
            res["price_eur_mwh"] = price
            records.append(res)

        out_df = pd.DataFrame(records).set_index("timestamp")

        tot_imp_mwh = float(out_df["p_import_mw"].sum())
        tot_exp_mwh = float(out_df["p_export_mw"].sum())
        tot_ch_mwh = float(out_df["p_charge_mw"].sum())
        tot_dis_mwh = float(out_df["p_discharge_mw"].sum())
        tot_curt_mwh = float(out_df["p_curtail_mw"].sum())
        tot_solar_mwh = float(out_df["solar_mw"].sum())
        tot_load_mwh = float(out_df["load_mw"].sum())

        tot_imp_cost = float(out_df["import_cost_eur"].sum())
        tot_exp_rev = float(out_df["export_revenue_eur"].sum())
        tot_deg_cost = float(out_df["degradation_cost_eur"].sum())
        net_total_cost = tot_imp_cost - tot_exp_rev + tot_deg_cost

        efc = (tot_ch_mwh + tot_dis_mwh) / (2.0 * self.cfg.energy_capacity_mwh)
        losses_mwh = tot_ch_mwh - tot_dis_mwh

        return SimulationResult(
            strategy_name=strategy_name,
            timeline=out_df,
            total_cost_eur=round(net_total_cost, 2),
            grid_import_cost_eur=round(tot_imp_cost, 2),
            grid_export_revenue_eur=round(tot_exp_rev, 2),
            degradation_cost_eur=round(tot_deg_cost, 2),
            total_grid_import_mwh=round(tot_imp_mwh, 2),
            total_grid_export_mwh=round(tot_exp_mwh, 2),
            total_solar_generated_mwh=round(tot_solar_mwh, 2),
            total_load_mwh=round(tot_load_mwh, 2),
            total_charged_mwh=round(tot_ch_mwh, 2),
            total_discharged_mwh=round(tot_dis_mwh, 2),
            total_curtailed_mwh=round(tot_curt_mwh, 2),
            equivalent_full_cycles=round(efc, 2),
            round_trip_losses_mwh=round(losses_mwh, 2),
            max_energy_balance_residual=round(max_residual, 6),
            constraint_violations=violations,
        )
