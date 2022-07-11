from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=150,
                  n_features=2,
                  centers=3,
                  cluster_std=0.5,
                  shuffle=True,
                  random_state=0)


print(X[:10])
print(y[:10])

# 앙상블을 사용하는 이유
#   일반화의 성능을 극대화하기 위해
#   (예측 성능의 분산을 감소시킬 수 있음)