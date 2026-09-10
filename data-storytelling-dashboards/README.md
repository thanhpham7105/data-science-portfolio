# Data Storytelling & Dashboards

## Overview
Designed an interactive Tableau dashboard to help a telecommunications company's leadership understand customer churn -- translating raw churn data into an executive-friendly visual narrative for a non-technical audience.

## Dashboard layout

**Top row -- KPIs.** Two headline numbers set the context immediately:
- **Overall churn rate (%)** -- the share of customers who cancel service.
- **Monthly recurring revenue at risk ($)** -- the recurring revenue tied to those cancellations.

**Middle row -- why customers churn.** Three bar charts break the KPI down into drivers:
- **Tenure band** -- short-tenure customers churn the most.
- **Payment method** -- manual vs. automatic payment users.
- **Contract type** -- month-to-month plans churn far more than longer-term contracts.

**Bottom -- where it's happening.** A U.S. map shades states by churn rate, so regional teams can see at a glance where retention effort matters most.

**Filters.** Two interactive filters (state, internet service type) let a viewer narrow the whole dashboard to a specific region or service line without needing to touch the underlying data.

## Adapting the story for different audiences

The same underlying data supports very different presentations depending on who's looking at it:

- **Level of detail.** A technical audience gets the underlying formula (e.g., churned customers ÷ total customers) and data-quality notes; a non-technical executive gets a plain-language read like "about a quarter of customers churned."
- **Visual complexity.** Technical stakeholders can layer in trend lines, parameters, and supporting tables; executives get a small set of KPIs, bars, and a map that directly answer the business question.
- **Terminology.** Technical framing references model metrics and cohorts; executive framing stays outcome-focused -- e.g., "encouraging autopay and longer contracts should reduce churn."

## Storytelling & accessibility design choices

- **Narrative flow** mirrors how executives actually reason through a problem: KPIs answer *what's happening*, the driver charts answer *why*, and the map answers *where to act*.
- **Visual emphasis**: large KPI numbers and clear titles draw the eye to what matters first.
- **Accessible color**: a color-blind-safe sequential palette, plus redundant data labels/tooltips so information isn't conveyed by color alone -- the dashboard still reads correctly in grayscale.

## Skills demonstrated
Dashboard design for non-technical audiences, KPI selection, interactive filtering, geographic visualization, accessibility-aware design, data-storytelling and audience-adaptation principles.
