# amazon-review-behavior-analysis
Analyzing how rating patterns (average, volatility, thresholds) predict future review volume: a Fixed Effects panel study on 7,500+ Amazon product-months.

# Do Star Ratings Predict Future Reviews? A Behavioral Analysis of 7,500+ Amazon Product Months
Platforms often assume that higher ratings drive more engagement, but does the shape of a rating matter more than the average itself? This project looks at whether a product's average rating, the consistency of that rating, and specific rating thresholds (like crossing 4.0 or 4.5 stars) can predict how many reviews a product will receive in the future.

## Motivation
Online platforms rely on reviews as a self reinforcing signal. More reviews build more trust, which drives more purchases, which in turn generates more reviews. But not every rating pattern generates the same future engagement. This project asks which characteristics of a product's rating history, whether that's its average, its consistency, or specific psychological thresholds, best predict how many reviews it will receive next month.

## Data
The dataset comes from Amazon Reviews 2023 (All Beauty category), restructured into a panel of 100 products observed over approximately 75 months, giving more than 7,500 product month observations. For each product month, I tracked the cumulative average rating, cumulative rating variance, monthly review count, cumulative review count, and two threshold indicators marking whether a product had crossed 4.0 or 4.5 stars.

## Method
Since the same 100 products are observed repeatedly over time, I used a Fixed Effects panel regression. This method controls for everything unique and constant about each product, like brand reputation or baseline popularity, and isolates how changes in rating characteristics relate to changes in future review volume within the same product over time.

I chose Fixed Effects over simpler alternatives, such as Pooled OLS or Random Effects, after a Hausman test confirmed that unobserved product specific factors were correlated with the predictors. Ignoring that correlation would have biased the results.

## Findings
A product's review count in the current month turned out to be the strongest predictor of next month's volume, suggesting that momentum matters more than reputation.

Products with more variance in their ratings, meaning a mix of very positive and very negative reviews, attracted more future engagement rather than less. Controversy or inconsistency in ratings may actually drive more attention.

Surprisingly, the average rating on its own was not a statistically significant predictor of future review volume once other factors were controlled for. This suggests platforms shouldn't assume that a higher average rating alone leads to more future engagement.

## Business Implications
These findings have some practical implications for platforms. Since a product's recent review activity is the strongest predictor of future review volume, platforms could identify products gaining early momentum and design small nudges, like timely follow up prompts, to help sustain that engagement.

The volatility finding is more counterintuitive. Products with inconsistent ratings actually attract more future reviews, not fewer. This challenges the common assumption that platforms should reduce visibility for lower rated or more divisive products.

Finally, relying on average rating alone as a health metric for a product can be misleading. Once momentum and volatility are accounted for, the average rating is a weaker signal than expected.

## Limitations
The model uses current month review count to help predict next month's review count, which introduces a known issue in dynamic panel models called Nickell bias. Since each product in this dataset is observed over a fairly long period, around 75 months on average, the practical impact of this bias is likely small, though a dynamic panel method like Arellano Bond would address it more directly.

These findings are also based only on the Beauty category, so the patterns might look different in categories with different purchase cycles, such as electronics versus everyday consumables.
