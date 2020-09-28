# Python入門とPythonによる機械学習

## Pythonの基礎

+ [basic_python.ipynb](https://github.com/namoshi/ml_intro/blob/master/basic_python.ipynb "basic_python.ipynb") --- pythonの基礎
+ [basic_numpy.ipynb](https://github.com/namoshi/ml_intro/blob/master/basic_numpy.ipynb) --- numpyの基礎（ベクトルと行列の計算）
+ [plot_python.ipynb](https://github.com/namoshi/ml_intro/blob/master/plot_python.ipynb) --- pythonでのグラフの作成

## ベイズ識別

+ [bayes_iris_2d.ipynb](https://github.com/namoshi/ml_intro/blob/master/bayes_iris_2d.ipynb)  --- Fisherのアヤメのデータのベイズ識別（ガクの長さと幅の2次元の特徴を使用，各クラスの条件付確率分布として正規分布を仮定し事後確率を推定（2次識別関数））
+ [Bayes_iris.ipynb](https://github.com/namoshi/ml_intro/blob/master/Bayes_iris.ipynb) ---  Fisherのアヤメのデータのベイズ識別（ガクの長さと幅，花びらの長さと幅の4次元特徴を使用，各クラスの条件付確率分布として正規分布を仮定し事後確率を推定（2次識別関数）） 
+ [knn-iris_2d.ipynb](https://github.com/namoshi/ml_intro/blob/master/knn-iris_2d.ipynb) --- K-近傍法によるアヤメのデータのベイズ識別（ガクの長さと幅の2次元の特徴を使用）
+ [gmm-iris_2d.ipynb](https://github.com/namoshi/ml_intro/blob/master/gmm-iris_2d.ipynb) --- Gaussian Mixture Modelによるアヤメのデータの識別（ガクの長さと幅の2次元の特徴を使用，各クラスの条件付確率分布をGauusian Mixture Modelで推定し，事後確率を計算）


## 予測のための線形モデル

+ [Regression-sklern](https://github.com/namoshi/ml_intro/blob/master/Regression-sklearn.ipynb)  ---  線形回帰の例（収率データの単回帰分析，マンション価格の重回帰分析，ボール投げの予測の重回帰分析，サイン関数の多項式回帰）
+ [ball_reg.ipynb](https://github.com/namoshi/ml_intro/blob/master/ball_reg.ipynb) --- 重回帰分析によるボール投げの予測（OLSを利用）
+ [poly_sin_sample.ipynb](https://github.com/namoshi/ml_intro/blob/master/poly_sin_sample.ipynb)  ---  サイン関数の多項式回帰


## 識別のための線形モデル

+ [perceptron_iris_2d.ipynb](https://github.com/namoshi/ml_intro/blob/master/perceptron_iris_2d.ipynb)  ---  Perceptronによるアヤメのデータ（2クラス，2次元）の識別
+ [adaline_iris_2d.ipynb](https://github.com/namoshi/ml_intro/blob/master/adaline_iris_2d.ipynb)  ---   ADALINEによるアヤメのデータ（2クラス，2次元）の識別
+ [logit_iris_2d.ipynb](https://github.com/namoshi/ml_intro/blob/master/logit_iris_2d.ipynb) --- ロジスティック回帰によるアヤメのデータ（2クラス，2次元）の識別


## 汎化性能

+ [poly_sin_sample.ipynb](https://github.com/namoshi/ml_intro/blob/master/poly_sin_sample.ipynb)  ---  サイン関数の多項式回帰（サンプル数と汎化性の関係）
+ [poly_sin_aic.ipynb](https://github.com/namoshi/ml_intro/blob/master/poly_sin_aic.ipynb)  ---  サイン関数の多項式回帰（情報量基準（AIC）によるモデル選択）
+ [RigeRegression_House.ipynb](https://github.com/namoshi/ml_intro/blob/master/RigeRegression_House.ipynb)  ---  ボストンの家の価格データのリッジ回帰
+ [Lasso_House.ipynb](https://github.com/namoshi/ml_intro/blob/master/Lasso_House.ipynb)  ---  ボストンの家の価格データのLasso
+ [SVM_iris_2d.ipynb](https://github.com/namoshi/ml_intro/blob/master/SVM_iris_2d.ipynb)  ---  サポートベクターマシン（SVM)のアヤメのデータへの適用
+ [lasso_model_selection.ipynb](https://github.com/namoshi/ml_intro/blob/master/lasso_model_selection.ipynb)  ---  モデル選択（Lasso）

## 情報抽出手法

+ [PCA_seiseki.ipynb](https://github.com/namoshi/ml_intro/blob/master/PCA_seiseki.ipynb)  ---  成績データの主成分分析（Principal Component Analysis）
+ [PCA_iris.ipynb](https://github.com/namoshi/ml_intro/blob/master/PCA_iris.ipynb)  ---  アヤメのデータの主成分分析（Principal Component Anaysis）
+ [LDA_iris.ipynb](https://github.com/namoshi/ml_intro/blob/master/LDA_iris.ipynb)  ---  アヤメのデータの線形判別分析（Linear Discriminant Analysis）
+ [CCA,ipynb](https://github.com/namoshi/ml_intro/blob/master/CCA.ipynb "CCV.ipynb") --- 正順相関分析(Canonical Correlation Analysis）

## カーネル学習

+ [poly_sin_KernelRidge.ipynb](https://github.com/namoshi/ml_intro/blob/master/poly_sin_KernelRidge.ipynb)  ---  カーネルリッジ回帰（サイン関数）
+ [kernel_logit.ipynb](https://github.com/namoshi/ml_intro/blob/master/kernel_logit.ipynb)  ---  カーネルロジスティック回帰（サークルデータ）
+ [kernel_pca.ipynb](https://github.com/namoshi/ml_intro/blob/master/kernel_pca.ipynb)  ---  カーネル主成分分析（サークルデータ）

## クラスタリング

+ [HClust_demo.ipynb](https://github.com/namoshi/ml_intro/blob/master/HClust_demo.ipynb)  ---    階層的クラスタリングのデモ
+ [HClust_digit.ipynb](https://github.com/namoshi/ml_intro/blob/master/HClust_digit.ipynb)  ---  階層的クラスタリング（数字データ）
+ [Kmeans_demo.ipynb](https://github.com/namoshi/ml_intro/blob/master/Kmeans_demo.ipynb)  ---  K-means法のデモ
+ [Kmeans_digit.ipynb](https://github.com/namoshi/ml_intro/blob/master/Kmeans_digit.ipynb)  ---  K-means法（数字データ）
+ [Kmeans_color_quantization.ipynb](https://github.com/namoshi/ml_intro/blob/master/Kmeans_color_quantization.ipynb)  ---  K-Means法（色空間のクラスタリング（色の量子化））

## ディープラーニング

+ [convolution.ipynb](https://github.com/namoshi/ml_intro/blob/master/convolution.ipynb)  ---  1次元のコンボリューションの例



