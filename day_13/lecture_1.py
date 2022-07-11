#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 15:07:58 2022

@author: youngmin
"""

# 비지도 학습
# 지도 학습 / 비지도 학습 / 준지도 학습 / 강화 학습
# 지도 학습과 다르게 종속변수(정답데이터)가 제공되지 않는 데이터에 대한 학습을 처리하는 기법

# 비지도학습의 결과는 주관적인 판단으로 처리

# 비지도 학습의 카테고리
# - 차원 축소
#   데이터에 포함된 특성 중 유의미한 값을 추출
#   데이터에 포함된 특성을 투영하여 대표하는 값을 반환
#   (시각화를 위해서 사용되는 경우가 많음)
#   (소셜 네트워크 분석, 이미지 분석)

# - 군집 분석
#   데이터(샘플)의 유사성을 비교하여 동일한 특성으로 구성된 데이터(샘플)들을 하나의 군집으로 처리하는 기법

# 비지도 학습 시 유의사항
#   비지도 학습의 결과는 100% 신뢰할 수 없음
#   매번 실행될 때마다 결과는 변경될 수 있음

# 군집 분석을 활용하여 데이터의 클러스터링 처리과정을 확인

# 지도학습에 군집 분석의 결과를 활용하는 방법

from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=150,
                  n_features=2,
                  centers=3,
                  cluster_std=0.5,
                  shuffle=True,
                  random_state=0)

print(X[:10])
print(y[:10])

import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1], c='white', marker='o', edgecolors='black', s=50)

plt.grid()
plt.show()

# 최근접 이웃 알고리즘
# 군집분석을 위한 클래스
# KMeans
# - 가장 많이 사용되는 군집분석 클래스
# - 알고리즘이 단순하고 변경에 용이
#   (수정 사항의 반영이 손쉬움)

from sklearn.cluster import KMeans

# 최적의 군집(클러스터)의 개수를 검색
# - 엘로우 방법을 활용하여 처리할 수 있음
values = []
for i in range(1, 11) : 
    
    km = KMeans(n_clusters=i,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=0)
    km.fit(X)
    # 클러스터 내의 각 클래스의 SSE 값을 반환하는
    # inertia_ 속성값 
    values.append(km.inertia_)
print(values)

plt.plot(range(1,11), values, marker='o')
plt.xlabel('numbers of cluster')
plt.ylabel('inertia_')
plt.show()
# 각도 봐서 한번에 확 떨어지다 완만해짐 -> 3개정도 최적의 군집(엘로우기법)

# km = KMeans(n_clusters=3,
#             init='random',
#             n_init=10,
#             max_iter=300,
#             random_state=0)
# 군집 분석 알고리즘인 KMeans의 fit 메소드의 동작
# - n_clusters에 정의된 개수만큼 포인트를 지정하여 최적의 위치를 찾도록 검색하는 과정을 수행
km.fit(X)

# 군집의 결과를 생성하여 반환
y_cluster = km.predict(X)
print(y_cluster)

plt.scatter(X[y_cluster == 0, 0],
            X[y_cluster == 0, 1],
            s=50, c='Lightgreen',
            marker='s', label='Cluster')

plt.scatter(X[y_cluster == 1, 0],
            X[y_cluster == 1, 1],
            s=50, c='orange',
            marker='o', label='Cluster 2')

plt.scatter(X[y_cluster == 2, 0],
            X[y_cluster == 2, 1],
            s=50, c='red',
            marker='*', label='Cluster')

plt.legend()
plt.grid()
plt.show()


# plt.scatter(X[y_cluster == 0, 0],
#             X[y_cluster == 0, 1],
#             s=50, c='Lightgreen',
#             marker='s', label='Cluster')

# plt.scatter(X[y_cluster == 1, 0],
#             X[y_cluster == 1, 1],
#             s=50, c='orange',
#             marker='o', label='Cluster 2')

# plt.scatter(X[y_cluster == 2, 0],
#             X[y_cluster == 2, 1],
#             s=50, c='red',
#             marker='*', label='Cluster')


# 군집이 늘어날수록 오차는 작아질수밖에 없음














