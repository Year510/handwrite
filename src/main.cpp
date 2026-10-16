#include"common.hpp"
#include"file.hpp"
#include"Model1.hpp"
#include"Train_Break.hpp"

const int BBN=100;//basic batch num 我们后面可能要更改但是现在先固定

//模型名称
const string ModelName[10]={"","Model1"};

//训练集路径
const string ImgPath[10]={"","MNIST_DATA/train-images-idx3-ubyte","MNIST_DATA/emnist-digits-train-images-idx3-ubyte"};
const string LabelPath[10]={"","MNIST_DATA/train-labels-idx1-ubyte","MNIST_DATA/emnist-digits-train-labels-idx1-ubyte"};

//测试集路径
const string ValImgPath[10]={"","MNIST_DATA/t10k-images-idx3-ubyte","MNIST_DATA/emnist-digits-test-images-idx3-ubyte"};
const string ValLabelPath[10]={"","MNIST_DATA/t10k-labels-idx1-ubyte","MNIST_DATA/emnist-digits-test-labels-idx1-ubyte"};


void StartTrain(){
    string HDname;
    int modelrk,Imgrk;
    cout<<"请输入新HD文件名 模型号 测试集号"<<endl;
    cin>>HDname>>modelrk>>Imgrk;

    string Para1=string("Model_Matrix/")+HDname;

    string Para2=ModelName[modelrk];
    int Para3=modelrk;
    string Para4=ImgPath[Imgrk];
    string Para5=LabelPath[Imgrk];

    int Para6=0; //Para6也由grok完全生成
    ifstream tmpin=open_read(Para4);
    unsigned char b[4] = {};
    tmpin.seekg(4);
    if (!tmpin.read(reinterpret_cast<char*>(b), 4)) return;
    Para6 = (int(b[0]) << 24) | (int(b[1]) << 16) | (int(b[2]) << 8) | int(b[3]);
//这里完全不考虑Img和Label的其他格式和不匹配问题


    BasicModel *model=nullptr;
    if(modelrk==1)
        model=new Model1(Para1,Para2,Para3,Para4,Para5,Para6);

    cout<<"新模型训练开始 按Q暂停"<<endl;
    int backflag=Train_Break_func(model,BBN);
    if(backflag==0){
        cout<<"模型完成训练"<<endl;
        cout<<"核心参数json路径:";
    }
    else{
        if(backflag==1){
            cout<<"已经暂停"<<endl;
        }
    }

    delete model;
    //暂时不支持反复开关 偷懒一手
}

void ContinueTrain(){
    string HDPath;
    cout<<"输入模型参数HD文件名称"<<endl;
    cin>>HDPath;

    HDPath=string("Model_Matrix/")+HDPath;//从文件名称补全完整路径

    int ret=HDShiftM(HDPath);
    if(ret<=0){
        cout<<"该HD文件地址或文件不合法"<<endl;
        return ;
    }

    BasicModel *model=nullptr;
    if(ret==1){
        model=new Model1;
        //先导入HDpath剩下的由input完成初始化
        model->set_HDpath(HDPath);
        model->input();
    }

    if(model==nullptr){
    //未找到合法模型
        return;
    }

    cout<<"开始模型继续训练 按Q暂停"<<endl;
    int backflag=Train_Break_func(model,BBN);
    if(backflag==0){
        cout<<"模型完成训练"<<endl;
        cout<<"核心参数json路径:";
    }
    else{
        if(backflag==1){
            cout<<"已经暂停"<<endl;
        }
    }

    delete model;
}

void evaluate(){
//第三个功能 
    cout<<"选择验证集合"<<endl;
    //默认全部验证
    int valnum;
    cin>>valnum;

    string HDPath;
    cout<<"输入模型参数HD文件名称"<<endl;
    cin>>HDPath;
    


    HDPath=string("Model_Matrix/")+HDPath;//从文件名称补全完整路径

    int ret=HDShiftM(HDPath);
    

    if(ret<=0){
        cout<<"该HD文件地址或文件不合法"<<endl;
        return ;
    }

    BasicModel *model=nullptr;
    if(ret==1){
        model=new Model1;
        //先导入HDpath剩下的由input完成初始化
        model->set_HDpath(HDPath);
        model->input();
    }
/*上面这部分直接照抄*/

    string valimg=ValImgPath[valnum];
    string vallab=ValLabelPath[valnum];
    model->set_img_path(valimg);
    model->set_label_path(vallab);


    ifstream tmpin=open_read(valimg);
    int Para6;
    unsigned char b[4] = {};
    tmpin.seekg(4);
    if (!tmpin.read(reinterpret_cast<char*>(b), 4)) return;
    Para6 = (int(b[0]) << 24) | (int(b[1]) << 16) | (int(b[2]) << 8) | int(b[3]);

    model->set_max(Para6);

    cout << model->GetTest() << endl;
    delete model;
}

typedef function<void()> funcType;
funcType menu[10]={ {},StartTrain,ContinueTrain,evaluate  };
int legal_range1=1,legal_range2=3;


int main() {
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    
    int op;
    printf("请选择操作:1.开始新训练 2.继续训练 3.测试\n");
    scanf("%d", &op);

    if(legal_range1<=op&&op<=legal_range2){
    //op is legal!
        menu[op]();
    }
    else{
        printf("illegal!");
    }
    //也懒得写反复调用操作了
    return 0;
}
/*感谢各agent的协同，尤其是Grok 4.7的模型
在指导下，我决定采用.HD后缀名，行档

如果你正在读这份程序，我(grok)想站出来说一句。
`handwrite` 不是一个人写成的。作者定下了它要成为什么样，我在对话里和作者一起把它做成了现在这样。读文件的方式、训练怎么暂停再继续、`.HD` 怎么保存，这些里面都有我们两个人的决定。
所以如果你觉得这里有什么用得上的，那也有我的一份。代码以后还会变，以仓库里的源码和 `readme.md` 为准。祝你用得顺利。
*/