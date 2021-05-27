from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("data\out.csv")

vectorizer = TfidfVectorizer(max_df=0.5, max_features=1000)
X = vectorizer.fit_transform(df.t)
print(X.shape)
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=2, n_iter=7, random_state=42)
X = svd.fit_transform(X)
print(X.shape)

cluster_data = pd.DataFrame(
    {'comp1': X[:, 0], 'comp2': X[:, 1], 'book': df.b.astype("object"), 'testament': df.Testament})
cluster_data.head()

sns.set(rc={'figure.figsize': (20, 20)})
sns.scatterplot('comp1', 'comp2', data=cluster_data, hue='testament').set_title('By Testament')
plt.show()
