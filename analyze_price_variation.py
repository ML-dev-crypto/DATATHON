import pandas as pd

df = pd.read_csv('product_sales_data.csv')

print('=' * 50)
print('PRICE VARIATION ANALYSIS')
print('=' * 50)

# Overall price stats
print('\nOverall Price Statistics:')
print(f'  Min: ${df["price"].min():.2f}')
print(f'  Max: ${df["price"].max():.2f}')
print(f'  Mean: ${df["price"].mean():.2f}')
print(f'  Std: ${df["price"].std():.2f}')
cv = df["price"].std() / df["price"].mean()
print(f'  CV (Std/Mean): {cv:.2%}')

# Price variation within each product
print('\nPrice Variance per Product (unique prices per product):')
price_var = df.groupby('product_id')['price'].agg(['nunique', 'std', 'mean', 'min', 'max'])
price_var['cv'] = price_var['std'] / price_var['mean']
print(price_var.sort_values('nunique', ascending=False).head(10))

num_price_changes = (price_var['nunique'] > 1).sum()
num_fixed = (price_var['nunique'] == 1).sum()
print(f'\nProducts with price changes: {num_price_changes}')
print(f'Products with fixed price: {num_fixed}')

# Correlation analysis
print('\nCorrelation with units_sold:')
numeric_cols = ['price', 'discount_percent', 'is_promoted', 'stock_available', 'rating']
for col in numeric_cols:
    corr = df['units_sold'].corr(df[col])
    print(f'  {col}: {corr:.3f}')

# Category-wise price variation
print('\nPrice Variation by Category:')
cat_price = df.groupby('category')['price'].agg(['mean', 'std', 'min', 'max'])
cat_price['cv'] = cat_price['std'] / cat_price['mean']
print(cat_price)
