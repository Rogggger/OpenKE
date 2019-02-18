# OpenKE
这是清华OpenKE的一个fork，方便直接对TransE、TransH、TransR进行训练获得triple classification的结果。

更多用法详见[这里](https://github.com/thunlp/OpenKE/)

若要使用其它数据集，请按原操作说明的方式准备好train2id.txt、test2id.txt和vaild2id.txt这三个文件放入benchmark/<yourdataset>下，数据集的名称请**全大写**，再从benchmark/FB15K下将n-n.py文件拷贝至此文件夹下运行，获得n-n.txt等。

对应的TransE、TransH、TransR直接用python运行根目录下的对应train文件即可得到训练和测试的结果（只包含triple classification）。

运行方式为（其中fb15k为数据集名称，会自动转换成大写）:
~~~
python train_transe.py fb15k
~~~

