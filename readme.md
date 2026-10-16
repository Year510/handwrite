这是一个 10 类、28×28 灰度图的训练程序。训练集和测试集都是 IDX 字节文件：图像文件头 16 字节，其后每张图 784 字节；标签文件头 8 字节，其后每个标签 1 字节。第 n 张图和第 n 个标签是一对。张数是文件头偏移 4 处的 4 字节大端整数。

数据集放在 `MNIST_DATA`，只留在本机，不放进 Git。现在有两套，菜单里的数据集号是数组下标：

- `1`：MNIST 手写数字。训练 `train-images-idx3-ubyte`、`train-labels-idx1-ubyte`，各 60000。测试 `t10k-images-idx3-ubyte`、`t10k-labels-idx1-ubyte`，各 10000。
- `2`：EMNIST Digits。训练 `emnist-digits-train-images-idx3-ubyte`、`emnist-digits-train-labels-idx1-ubyte`，各 240000。测试 `emnist-digits-test-images-idx3-ubyte`、`emnist-digits-test-labels-idx1-ubyte`，各 40000。

训练中可以按 Q 暂停。完成或暂停都会把进度写成 `.HD` 行档，下次从停下的张数继续。源码和文本使用 UTF-8。IDX 和行档都按二进制打开。

当前程序在 `src`。`CMakeLists.txt` 只编译 `src/main.cpp`。`src2` 是早先一版 `src` 由当时的 Grok 补全后留下来的，不参与这次编译，只作历史保留。生成目录是 `build`（MinGW g++，C++17）。

第一次配置，在仓库根目录执行：

```powershell
cmake -S . -B build -G "MinGW Makefiles"
cmake --build build
.\build\handwrite.exe
```

`build` 还在时，以后每次是：

```powershell
cmake --build build
.\build\handwrite.exe
```

## 目录

MNIST_DATA
本地数据，只读。程序用相对路径访问，例如 `MNIST_DATA/train-images-idx3-ubyte`。`FilePath.hpp` 的 `open_read` 和 `open_write` 把这种路径变成绝对路径并按二进制打开。项目根目录靠这里的 `train-images-idx3-ubyte` 识别，所以这份 MNIST 训练图像要留在本机。

Model_Matrix
每个模型一个 `.HD` 行档。新训练时文件名由用户输入，相对路径是 `Model_Matrix/` 加上这个文件名。

src
正在使用的源码。函数体写在头文件里。

src2
旧版源码的补全，当前构建不会读它。

## 行档 .HD

一行条目，下一行是值。空行跳过。`bool` 写成 `0` 或 `1`。公共条目读到 `Specific Model data:` 为止，这一行之后是派生类自己的参数。

公共条目：

- `model_name`
- `model_name_rk`
- `train_finished`
- `train_img_path`
- `train_label_path`
- `train_img_num`
- `max_train_num`
- `Specific Model data:`

`model_name_rk` 是模型号。继续训练时 `HDShiftM` 读出这个整数，用它决定创建哪个派生类。模型 1 的值是 `1`。

`Model1` 在标记行之后写一行 `weight`，再写 10×784 个权重。读入时先读掉 `weight` 这一行，再读数字。权重用 `numeric_limits<double>::max_digits10` 位有效数字写出，使读回的 `double` 与写出前相同。

成员 `HDpath` 保存这个 `.HD` 的相对路径。新训练由带参构造函数赋值。继续训练先 `set_HDpath`，再 `input`。离开训练时写出用的就是它。文件流用完即关，不放在对象里。

## 一次运行怎么走

菜单是 `1` 开始新训练，`2` 继续训练，`3` 在测试集上计算正确率。batch 固定为 `100`。一次启动只做一件事。

开始训练：用户给出新行档文件名、模型号、数据集号。模型号 `1` 对应 `Model1`。数据集号 `1` 或 `2` 选上面两套训练集。`main` 从图像文件头读出张数，然后走带参构造：

`Model1(HDpath, model_name, model_name_rk, 图像路径, 标签路径, 张数)`

这个构造先调用基类的同参数构造，记下路径、名字、模型号，并把 `train_img_num` 设为 0、`train_finished` 设为 0、`max_train_num` 设为读到的张数。`Model1` 再把权重初始化为种子 1 的 `[-0.01, 0.01]`。

继续训练：用户给出已有 `.HD` 的文件名，程序补上 `Model_Matrix/`。`HDShiftM` 只读出 `model_name_rk`。值为 `1` 时 `new Model1()`，这是无参构造，不随机初始化。然后 `set_HDpath`，再调用无参 `input()`，用行档填入路径、进度和权重。

两条训练路都在对象准备好之后调用 `Train_Break_func(model, 100)`。`Train_Break` 只持有 `BasicModel*`。每一轮把 batch 交给 `train`。`train_img_num` 记的是已经训练的图片张数。剩余图片不足一个 batch 时，只读剩下的那些。张数达到 `max_train_num` 时 `train_finished` 设为 1。

`train` 的返回值：

- `0`：这一批正常，继续循环
- `1`：这是最后一批，训练完成
- `3`：开始时就已经完成，这次 `train` 不生效

`CoreTrain` 把一切非 `0` 都当成这一轮训练结束。`Train_Break_func` 返回 `0` 表示训练完成，返回 `1` 表示按下 Q。焦点在当前终端时 Q 才有效。两条离开路径都先 `EndTrain`，再 `join` 监察线程。`EndTrain` 调用无参 `output()`，写出 `.HD`。一次启动里，训练集可以读许多次；继续训练时 `.HD` 读一次；写出只有离开这一次。

测试：用户给出数据集号和已有 `.HD` 文件名。程序读入模型，把图像和标签路径换成该编号的测试集，再从测试图像文件头读出张数。`GetTest` 每次读 100 张，判完就释放，正确率是判对的张数除以测试集张数。

`solve` 返回 `possible`，是单张图片的 10 类概率。`Model1` 每个类别的得分是各像素上 `cosh(权重 × 像素)` 的和，再做 softmax。学习率 `0.01`，梯度按批平均。

## input 和 output

多模型通过 `BasicModel*` 调用，所以无参的 `input()` 和 `output()` 是纯虚函数，基类和派生类签名相同。运行时进入派生类。

派生类打开 `HDpath` 一次，再点名调用基类里带流的 `input(ifstream&)` 或 `output(ofstream&)`。这两个函数只处理公共字段，读到或写到 `Specific Model data:` 为止，不自己打开文件。派生类接着用同一个流读写自己的参数。带流的函数可以和虚函数同名，它们是重载，不参与这次多态。

## 各文件现在的职责

`FilePath.hpp`
`open_read`、`open_write` 把相对路径变成绝对路径并打开，都使用二进制模式。

`common.hpp`
系统头文件和项目里使用的 `std` 短名字集中在这里。`BasicModel` 保存路径、进度和张数。无参构造留给继续训练，实际数据由 `input` 写入。带参构造留给新训练。带流的 `input` / `output` 读写公共字段。`HDShiftM` 从行档读出 `model_name_rk`，失败返回 `-1`。

`file.hpp`
`inputImg`、`inputLabel` 按路径调用 `open_read`，跳过文件头，读出一批图片或标签。图片像素除以 255。数组由调用方 `delete[]`。

`Model1.hpp`
菜单里的模型 1。10×784 的权重。`train` 由 `Model1` 整段实现。`GetTest` 按批计算测试集正确率。

`Train_Break.hpp`
监察线程看 Q。`EndTrain` 调用 `output()`。完成和暂停都会调用它，并 `join`。

`main.cpp`
菜单、新训练、继续训练和测试。数据集路径写在四个数组里，下标 `1` 是 MNIST，下标 `2` 是 EMNIST Digits。
