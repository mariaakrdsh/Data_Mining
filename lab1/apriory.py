import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
import networkx as nx

data_path = 'Groceries_dataset.csv'
groceries_data = pd.read_csv(data_path)
groceries_data.dropna(inplace=True)

groceries_data['Transaction_ID'] = groceries_data['Member_number'].astype(str) + '-' + groceries_data['Date']
one_hot_encoded = pd.get_dummies(groceries_data.set_index('Transaction_ID')['itemDescription']).reset_index()
aggregated = one_hot_encoded.groupby('Transaction_ID').max().reset_index(drop=True)

item_frequency = aggregated.sum().sort_values(ascending=False)
plt.figure(figsize=(10, 8))
sns.barplot(x=item_frequency.head(20), y=item_frequency.head(20).index)
plt.title('Top 20 Most Frequent Items')
plt.xlabel('Frequency')
plt.ylabel('Items')
plt.show()

frequent_itemsets = apriori(aggregated, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.01)
rules = rules.sort_values('lift', ascending=False)

top_rules = rules.head(10)
plt.figure(figsize=(12, 8))
sns.barplot(x=top_rules['lift'], y=top_rules['antecedents'].astype(str) + ' -> ' + top_rules['consequents'].astype(str))
plt.title('Top 10 Association Rules by Lift')
plt.xlabel('Lift')
plt.xticks(rotation=45)
plt.show()

G = nx.from_pandas_edgelist(top_rules, source='antecedents', target='consequents', edge_attr=True)
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=0.5)
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2500, edge_color='k', linewidths=1, font_size=12, arrows=True)
edge_labels = nx.get_edge_attributes(G, 'lift')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title('Network Graph of Association Rules')
plt.show()