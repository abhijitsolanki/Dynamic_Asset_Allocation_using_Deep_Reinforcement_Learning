# **📊 Dynamic Asset Allocation using Deep Reinforcement Learning**

This project focuses on building an intelligent portfolio management
system that dynamically allocates assets on a daily basis to **maximize
returns while minimizing risk**, especially during uncertain market
conditions.

The model leverages **Deep Reinforcement Learning (DRL)** to learn
optimal investment strategies from historical market data.

## **🚀 Project Overview**

-   Designed a system for **dynamic portfolio allocation**

-   Used **Deep Deterministic Policy Gradient (DDPG)** to handle
    > continuous asset weights

-   Built a custom **trading environment** incorporating:

    -   Market volatility

    -   Portfolio constraints

    -   Investor risk preferences

-   Enabled the model to adapt its strategy based on changing market
    > conditions

## **🧠 Model Architecture**

-   **Actor Network\
    > **Determines the optimal allocation of capital across assets

-   **Critic Network\
    > **Evaluates how good the chosen allocation is using a Q-function

-   **Exploration Strategy\
    > **Initially explores using noise-based actions and gradually
    > shifts toward optimal decisions

-   **Reward Mechanism\
    > **Uses a performance baseline to improve stability and reward
    > optimization

## **🌍 Asset Selection (User Input)**

-   Users can select **up to 10 assets** from **any global market**

-   The system supports:

    -   Indian equities

    -   US stocks

    -   Crypto assets

    -   Forex pairs

This flexibility allows the model to learn across **diverse markets and
asset classes**.

## **📊 Portfolio Behavior**

-   Allocations are updated **daily**

-   The model:

    -   Increases exposure during favorable conditions

    -   Reduces risk during volatility or downturns

-   Focuses on **risk-adjusted returns rather than just profits**

## **📉 Benchmark Comparison -- Nifty 50**

The model\'s performance is evaluated against the **Nifty 50 index**.

### **Comparison Insights:**

-   Measures **outperformance (alpha)** over the benchmark

-   Evaluates **risk exposure (beta)**

-   Analyzes **drawdowns and volatility**

-   Provides a clear understanding of how well the model performs in
    > real market conditions

## **📈 Performance (Test Period: Aug 2022 -- Mar 2024)**

-   **Cumulative Return:** 55.18%

-   **CAGR:** 20.16%

-   **Sharpe Ratio:** 2.4

-   **Sortino Ratio:** 3.37

-   **Annual Volatility:** 11.65%

-   **Max Drawdown:** -9.06%

-   **Alpha (vs Nifty 50):** 0.15

-   **Beta:** 0.81

## **🎯 Key Highlights**

-   Handles **continuous portfolio optimization**

-   Works in **real-world market conditions**

-   Balances **risk and return effectively**

-   Adapts dynamically during financial stress periods
