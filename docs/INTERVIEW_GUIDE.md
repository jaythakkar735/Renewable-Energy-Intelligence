# The Complete Plain-English Interview & Architecture Guide
## *Renewable Energy Forecasting & Battery Storage Optimization*

> **Target Audience**: BE IT / CS Graduates preparing for technical interviews, portfolio presentations, or system walkthroughs.  
> **Philosophy**: Zero buzzwords. Every complex math formula, physical constraint, and machine learning pipeline is explained in clear, intuitive, and practical terms.

---

## 1. The 30-Second Elevator Pitch
*(Memorize this for "Tell me about your most challenging project.")*

> "I built an end-to-end energy intelligence system that co-optimizes day-ahead solar power forecasting and industrial battery dispatch on 5.75 years of real Danish electricity grid data. 
> Using Linear Programming with the HiGHS solver, the system schedules a 10 MWh battery to minimize electricity procurement costs, saving over **€41,900 (5.56%)** across a 9-month holdout test set—capturing **94.8% of theoretical perfect foresight**. 
> Crucially, I performed a rigorous statistical savings decomposition with weekly block bootstrapping, discovering that 24-hour price arbitrage optimization drives over 95% of the economic value, while real-time solar forecast errors yield a modest net drag under asymmetric imbalance settlement."

---

## 2. The "Explain Like I'm Five" (ELI5) Analogy

Imagine you manage a large refrigerated factory:
1. **The Problem**: You need electricity 24 hours a day. You have solar panels on your roof, and you buy the rest from the power grid. But power grid prices change every hour—sometimes power is cheap (€10/MWh), sometimes expensive (€150/MWh), and sometimes wind generation is so high that power prices go **negative** (the grid literally pays you to consume power!).
2. **The Battery**: You install a giant rechargeable battery. You want to charge it when power is cheap or when your solar panels are producing extra energy, and discharge it when power is expensive to avoid buying from the grid.
3. **The Catch**: 
   - A battery isn't 100% efficient. If you put 100 kWh in, you only get 88 kWh out (88% Round-Trip Efficiency)—12 kWh is lost as heat.
   - You must decide your battery's schedule for tomorrow **today at 12:00 PM**, before tomorrow begins!
4. **The Machine Learning**: You build an AI model (LightGBM) to forecast how much solar power your roof will generate tomorrow so the optimizer can plan ahead.
5. **The Real-World Surprise**: We discovered that knowing the *price swings* and planning ahead with smart math (Linear Programming) gives you 95% of your savings. Even if you use a simple "yesterday's solar was like today" forecast, the battery still saves almost the exact same amount of money!

---

## 3. High-Level System Architecture

The project is built in 5 modular stages:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            1. DATA INGESTION                                │
│  Open Power System Data (OPSD) • Denmark West (DK_1) • 50,401 Hourly Rows   │
│  Flags 13 SCADA telemetry freezes • Strict UTC • Non-destructive caching    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    2. CUTOFF-SAFE FEATURE ENGINEERING                       │
│  Strict 12:00 UTC Gate Closure • Astronomical Spencer (1971) Ephemeris      │
│  Trailing 365-day causal capacity proxy Ct • Non-negative lags >= 24h/48h   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    3. MACHINE LEARNING & UNCERTAINTY                        │
│  Capacity-Normalized LightGBM, Random Forest, Ridge                         │
│  Hard night zeros (Elevation <= 0) • Split-Conformal Intervals • P10/P50/P90│
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     4. LINEAR PROGRAMMING OPTIMIZER                         │
│  Rolling 24h HiGHS Solver • Physical power balance • RTE (88%)              │
│  5 Strategies: No Battery, Rule-Based, LP+Naive, LP+ML, Perfect Foresight   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    5. EVALUATION & STREAMLIT DASHBOARD                      │
│  19 PyTest Invariant Tests • Weekly Block Bootstrap (B=1,000) 95% CIs       │
│  Light/Dark Theme • WCAG AA Contrast • Standalone HTML Executive Export     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Deep-Dive: The Core Technical Concepts Simplified

### Concept 1: Day-Ahead Market Gate Closure (12:00 UTC Cutoff)
- **What is it?** In European power markets (Nord Pool, EPEX Spot), electricity for tomorrow (hours 00:00 to 23:00 on Day $D+1$) is auctioned today (Day $D$). The auction closes sharply at **12:00 UTC**.
- **Why it matters for ML**: If your model predicts solar power for tomorrow at 2:00 PM, it cannot use sensor readings from 1:00 PM today! 
- **How we solved it**: For any hour $h$ on Day $D+1$, if $h < 12$, the minimum lag allowed is 24 hours. If $h \ge 12$, the minimum lag allowed is 48 hours. This guarantees **zero lookahead data leakage**.

### Concept 2: Solar Geometry & Night Zero Clamping
- **What is it?** Machine learning models don't know astronomy. A neural network or gradient booster might predict `0.3 MW` at 3:00 AM because of random noise in the training data.
- **Why it matters**: A commercial battery might try to plan charging around phantom nighttime solar generation.
- **How we solved it**: We implemented the **Spencer (1971) solar ephemeris** equations directly in Python to compute the exact solar elevation angle based on latitude and longitude ($56^\circ\text{N}, 9^\circ\text{E}$). When solar elevation $\le 0^\circ$ (night), the forecast is hard-clamped to **strictly 0.00 MW**.

### Concept 3: The -40 MW Daytime Bias & Capacity Normalization
- **The Mystery**: When we first trained models on raw megawatts (MW), the model under-predicted daytime solar generation by 40 MW on average!
- **The Investigation**: Why? We analyzed historical Danish grid records and discovered that Denmark West doubled its solar capacity (+107%, from 350 MW in 2015 to 727 MW in 2020). Models trained on earlier years were predicting for a much smaller solar fleet.
- **The Fix**: We created a **causal trailing 365-day capacity proxy**:
  $$C_t = \max_{k \ge 48} y_{t-k}$$
  We normalized the target: $y_{\text{norm}} = y / C_t$. This cut daytime bias in half (down to -20 MW) without looking into the future!

### Concept 4: Linear Programming vs. "Rule-Based" Heuristics
- **Rule-Based**: If solar generation > load, charge. If price is high, discharge. 
  - *Flaw*: It cannot "look ahead". It might discharge fully at 5:00 PM when price is €80/MWh, only to find price spikes to €250/MWh at 7:00 PM when the battery is already empty!
- **Linear Programming (LP)**: You write down the exact equations of what you want (minimize total cost) and your constraints (battery capacity, maximum inverter speed, efficiency losses). The **HiGHS solver** tests every possible schedule across all 24 hours simultaneously in 5 milliseconds, guaranteeing the absolute best mathematical schedule.

### Concept 5: Why Did the ML Solar Forecast Produce -€443 vs. Naive?
*(The #1 most interesting insight in the whole project)*
- Many data scientists assume that a higher ML skill score (+4.8% RMSE improvement) automatically creates more business value.
- **Here is what actually happened**:
  1. The battery's primary source of revenue is **price arbitrage** (charging at 3:00 AM at €15/MWh, discharging at 8:00 AM at €110/MWh).
  2. Day-ahead spot prices are already known with 100% certainty after 12:00 PM.
  3. When LightGBM slightly over-forecasts solar generation at noon tomorrow, the LP optimizer decides: *"Great, I don't need to grid-charge at 4:00 AM because solar will fill the battery."*
  4. Tomorrow arrives, clouds appear, actual solar is lower than predicted, and the facility is forced to buy expensive grid power in real-time.
  5. The simple **Last-Available-Day baseline** is smooth and conservative, avoiding aggressive misallocations.
- **Interview Takeaway**: In constrained optimization systems with efficiency losses, point forecast errors can cause asymmetric economic penalties.

---

## 5. Top 15 Technical Interview Questions & Scripted Answers

### Q1: What was your role in this project?
> **Answer**: "I designed, implemented, tested, and deployed the entire system end-to-end. That included building the data extraction and cleaning pipeline for 50,000 hourly records, engineering physical solar geometry features, training machine learning and quantile regressors, formulating the Linear Program in SciPy HiGHS, authoring 19 PyTest verification tests, and building the interactive Streamlit dashboard."

### Q2: Why did you choose Linear Programming instead of Dynamic Programming or Reinforcement Learning?
> **Answer**: "Because the battery physics and tariff costs form linear relationships. The battery state-of-charge transitions, power limits, and hourly grid costs are all linear equalities and inequalities. 
> Reinforcement learning requires millions of training steps, can hallucinate unfeasible actions, and offers no optimality guarantees. Linear Programming using modern dual-simplex solvers like HiGHS finds the provably global optimal schedule in under 10 milliseconds with zero constraint violations."

### Q3: How did you prevent data leakage in your time-series model?
> **Answer**: "I modeled the real-world operational gate closure of the Nord Pool day-ahead power market. Trades close at 12:00 UTC on day D for physical delivery across all 24 hours of day D+1. 
> Therefore, no feature could use any data newer than 12:00 UTC on day D. For midnight to noon on day D+1, this enforces a minimum 24-hour lag; for noon to midnight, it enforces a minimum 48-hour lag. I wrote an automated perturbation test in PyTest that mutates historical values and proves that predictions never change based on future rows."

### Q4: How do you handle round-trip efficiency in the battery?
> **Answer**: "Round-trip efficiency is 88%. That means charging efficiency is $\sqrt{0.88} \approx 93.8\%$ and discharging efficiency is $93.8\%$. In the state-of-charge equation:
> $$\text{SoC}_{t+1} = \text{SoC}_t + \frac{\eta_{\text{ch}} \cdot P_{\text{ch}, t} \cdot \Delta t}{E_{\text{cap}}} - \frac{P_{\text{dis}, t} \cdot \Delta t}{\eta_{\text{dis}} \cdot E_{\text{cap}}}$$
> This ensures that 12% of total energy throughput is correctly lost to heat and never counted as available energy."

### Q5: What happens when electricity prices go negative?
> **Answer**: "When high wind and low demand cause wholesale prices to turn negative (e.g., -€50/MWh), an unconstrained optimizer might try to 'burn' infinite energy by importing maximum grid power and dumping it to ground. 
> To prevent this unphysical pathology, I bounded solar curtailment to strictly available solar generation ($0 \le P_{\text{curt}} \le G$) and added complementary tolerances that prevent simultaneous charging and discharging."

### Q6: Why did you use LightGBM over Deep Learning (LSTM / Transformers)?
> **Answer**: "For tabular time-series features with explicit physical lags, calendar encodings, and ephemeris features, Gradient Boosted Decision Trees consistently match or outperform deep learning while training in 0.2 seconds on CPU rather than requiring GPU infrastructure. 
> In addition, LightGBM natively supports pinball loss for quantile regression (P10, P50, P90) and requires a fraction of the memory footprint."

### Q7: What is Conformal Prediction and why did you use it?
> **Answer**: "Standard machine learning models output a single point estimate without knowing their own uncertainty. 
> Split-conformal prediction is a distribution-free method that uses a held-out calibration set (the 2019 validation year) to compute non-conformity residuals. It provides a mathematical guarantee that 90% of future test observations will fall within the predicted upper and lower bounds without assuming normal distribution errors."

### Q8: What are SCADA telemetry freezes and how did you detect them?
> **Answer**: "In real grid sensors, telemetry feeds can freeze and repeat the exact same non-zero reading for dozens of hours due to communication failure. In DK_1, we identified 13 freeze events, including a 26-hour flatline at 284.59 MW on June 2–3, 2020. 
> If left untreated, models try to learn that weather stays constant for 26 hours. We flagged and masked these flatlines, shaded them in the dashboard, and excluded them from training loss functions."

### Q9: Why did you use Block Bootstrapping instead of standard Bootstrapping?
> **Answer**: "Standard bootstrapping assumes every data point is independent and identically distributed (i.i.d.). Electricity demand, solar output, and market prices have heavy autocorrelation across days and weeks. 
> If you sample individual hours randomly, you destroy the diurnal cycle. Weekly block bootstrapping resamples in continuous 168-hour blocks, preserving the real-world temporal correlation structure."

### Q10: How did you test your code?
> **Answer**: "I built a 19-test PyTest verification suite that checks:
> 1. Physical conservation of energy (First Law of Thermodynamics).
> 2. Zero negative values during solar nighttime.
> 3. Zero data leakage under feature perturbation.
> 4. Mathematical sum-consistency of the savings waterfall stages.
> 5. WCAG AA accessibility contrast compliance for all UI color tokens."

### Q11: Can you explain the Savings Waterfall stages?
> **Answer**: 
> 1. **No Battery (€754.5k)**: The facility buys all net power from the grid.
> 2. **Rule-Based (+€20.4k savings)**: Simple heuristic charging/discharging.
> 3. **LP + Naive (+€21.9k additional savings)**: 24h mathematical co-optimization.
> 4. **LP + ML (-€443 difference)**: The incremental impact of substituting the ML solar forecast.
> 5. **Perfect Foresight (+€2.3k remaining gap)**: The unachievable theoretical ceiling."

### Q12: Is a battery economically viable based on these numbers?
> **Answer**: "Based purely on day-ahead wholesale energy arbitrage, a 10 MWh battery saving ~€55,900/year annualized against a turnkey CAPEX of €300/kWh (€3.0M system) yields a payback period over 50 years. 
> Therefore, wholesale arbitrage alone is not viable. In the real world, batteries must stack revenues by participating in Frequency Containment Reserves (FCR), capacity markets, and shaving industrial peak-demand network tariffs."

### Q13: What software engineering practices did you follow?
> **Answer**: "I followed clean architecture principles: modular separation between core algorithms (`src/`), automated verification (`tests/`), configuration (`config.yaml`), and frontend views (`ui/pages/`). 
> I kept the raw 50k-row dataset strictly read-only and gitignored, pushed zero data binaries to GitHub, and added automated shell scripts (`scripts/run_all.sh`) for 1-command reproducibility."

### Q14: How does the Streamlit dashboard communicate with the backend?
> **Answer**: "The dashboard is stateless and reactive. When the user selects default parameters, it loads precomputed benchmark artifacts from `reports/` in milliseconds. 
> When the user modifies battery capacity or tariff sliders in the Battery Laboratory, the app calls `compare_dispatch_strategies()` to run a live HiGHS optimization across all 6,576 test hours in under 4 seconds, dynamically updating the KPI cards and waterfall charts."

### Q15: What would you do next if you had another month?
> **Answer**: "1. Incorporate Numerical Weather Prediction (NWP) satellite cloud-cover vectors to improve solar ramping accuracy.  
> 2. Add an intraday continuous trading layer to re-balance deviations between day-ahead schedules and real-time delivery.  
> 3. Model battery cell degradation non-linearly using Rainflow cycle-counting rather than a constant throughput penalty."

---

## 6. Project Key Numbers Quick-Reference

| Metric | Exact Value | Plain-English Meaning |
| :--- | :--- | :--- |
| **Grid Zone** | `DK_1` (Denmark West) | Jutland & Funen peninsula (Nord Pool market). |
| **Test Period** | Jan 1 – Sep 30, 2020 | 9 months / 6,576 out-of-sample hourly steps. |
| **Facility Scale** | 5.0 MW solar, 5.0 MW load | Standard medium industrial facility profile. |
| **Battery Rating** | 10 MWh capacity, 2.5 MW power | 4-hour duration ($C/4$ rate). |
| **Round-Trip Eff.** | 88% ($\approx 93.8\%$ each way) | 12% dissipated as thermal inverter/cell loss. |
| **Total Realized Savings** | **€41,931.98 (5.56%)** | Total reduction in net power procurement costs. |
| **Capture Rate** | **94.83%** | Percent of theoretical crystal-ball foresight achieved. |
| **Smarter LP Value** | **+€21,943.98** | Extra money saved by using math solver over rules. |
| **ML Forecast Value** | **-€442.66** | Difference between LightGBM and Naive baseline. |
| **95% Bootstrap CI** | `[-€899.98, -€70.48]` | Statistical range of the ML forecast value. |
| **Automated Tests** | **19 of 19 Passing** | 100% test pass rate across physical invariants. |
