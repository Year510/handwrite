#ifndef common_types_hpp
#define common_types_hpp


#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <thread>

#include <conio.h>
#include <windows.h>

#include "FilePath.hpp"

using std::atomic;
using std::cerr;
using std::cin;
using std::cout;
using std::endl;
using std::exp;
using std::function;
using std::getline;
using std::ifstream;
using std::numeric_limits;
using std::ofstream;
using std::printf;
using std::ref;
using std::scanf;
using std::setprecision;
using std::stoi;
using std::string;
using std::thread;
using std::uint8_t;



const int Image_Size = 28;

class possible {
    double p[10]{};
public:
    double& operator[](int index) { return p[index]; }
    double operator[](int index) const { return p[index]; }
};

class image {
    double color[Image_Size * Image_Size]{};
public:
    double& operator[](int index) { return color[index]; }
    double operator[](int index) const { return color[index]; }
};

// 派生类的 input / output 读写 Model_Matrix 里的 .HD文件

// 公共部分只记模型名和训练进度等，参数格式由派生类自己决定。
class BasicModel {
protected:
    string HDpath;//行档格式后缀名.HD
    //后续可能需要使用关于它的ifstream和ofstream

    string model_name;
    int model_name_rk;

	bool train_finished;
	string train_img_path;
	string train_label_path;
	int train_img_num; // 不存储batch 而是存储已经训练的img数量

    int max_train_num;

public:
    BasicModel(){
    //在开始训练或继续训练或导出模型前就必须完成全部构造，保证中间的训练过程全都是对外部无参调用的简单形式
    //只有读.HD才需要调用这里的构造，实际初始化由input完成
    //这里的函数体可以直接空置
    }

    BasicModel(string para1,string para2,int para3,string para4,string para5,int para6                
    ){//调用本构造函数必然是新模型训练，由main指定参数
        HDpath=para1;
        model_name=para2;
        model_name_rk=para3;
        train_finished=0;
        train_img_path=para4;
        train_label_path=para5;
        train_img_num=0;
        max_train_num=para6;
    }


    virtual ~BasicModel(){
        //似乎没有什么堆区内存需要释放
    }
    virtual possible solve(const image& input_img)=0;//纯虚函数
    virtual bool input() = 0;
    
    virtual bool input(ifstream &in){//需要返回值判断是否输入正确
        string tmpstr;
        while(1){
            if(!getline(in,tmpstr))return 1;
            if(tmpstr=="")continue;
            //switch(tmpstr){//这样写真的符合语法吗？
            
            bool FoundMatch=0;
            
            if(FoundMatch==0&&tmpstr=="model_name"){
                in>>model_name;
                FoundMatch=1;
            }
            if(FoundMatch==0&&tmpstr=="model_name_rk"){
                in>>model_name_rk;
                FoundMatch=1;
            }
            if(FoundMatch==0&&tmpstr=="train_finished"){
                in>>train_finished;
                FoundMatch=1;
            }
            if(FoundMatch==0&&tmpstr=="train_img_path"){
                in>>train_img_path;
                FoundMatch=1;
            }
            if(FoundMatch==0&&tmpstr=="train_label_path"){
                in>>train_label_path;
                FoundMatch=1;
            }
            if(FoundMatch==0&&tmpstr=="train_img_num"){
                in>>train_img_num;
                FoundMatch=1;
            }
            if(FoundMatch==0&&tmpstr=="max_train_num"){
                in>>max_train_num;
                FoundMatch=1;
            }
            if(FoundMatch==0&&tmpstr=="Specific Model data:"){
                FoundMatch=1;
                break;//后续的输入是具体派生类负责的
                
            }
            
            
            if(FoundMatch==0){
                return 1;//出错啦！
            }
            
        }
        return 0;
    }
    
    
    virtual bool output() = 0;
    virtual bool output(ofstream &out){
        out<<"model_name"<<endl;out<<model_name<<endl;
        out<<"model_name_rk"<<endl;out<<model_name_rk<<endl;
        out<<"train_finished"<<endl;out<<train_finished<<endl;//bool 应该输出的是0/1 而不是Fasle/Ture 反正这样输入输出都好写
        out<<"train_img_path"<<endl;out<<train_img_path<<endl;
        out<<"train_label_path"<<endl;out<<train_label_path<<endl;
        out<<"train_img_num"<<endl;out<<train_img_num<<endl;
        out<<"max_train_num"<<endl;out<<max_train_num<<endl;
        out<<endl;//华丽的换行
        out<<"Specific Model data:"<<endl;
        //剩下的就是不同模型的参数输出了，不是这个抽象类能做的
        return 0;
    }
    virtual int train(int batch)=0;//纯虚函数
    
    virtual double GetTest() = 0;

    void set_img_path(string tmpimg){
        train_img_path=tmpimg;
    }
    void set_label_path(string tmplabel){
        train_label_path=tmplabel;
    }
    void set_HDpath(string tmphd){
        HDpath=tmphd;
    }
    void set_max(int x){
        max_train_num=x;
    }
};

int HDShiftM(const string& PATH) {//该函数完全由grok生成
//打开失败、文件里没有 model_name_rk、下一行不是整数，返回 -1；否则返回那个模型编号
    ifstream in = open_read(PATH);
    if (!in.is_open()) return -1;
    string line;
    while (getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line != "model_name_rk") continue;
        if (!getline(in, line)) return -1;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        try {
            size_t pos = 0;
            int num = stoi(line, &pos);
            if (pos != line.size()) return -1;
            return num;
        } catch (...) {
            return -1;
        }
    }
    return -1;
}

possible Normal(const possible& x) {
    double m = x[0];
    for (int i = 1; i < 10; i++)
        if (x[i] > m) m = x[i];
    double tot = 0.0;
    possible ret;
    for (int i = 0; i < 10; i++) {
        ret[i] = exp(x[i] - m);
        tot += ret[i];
    }
    for (int i = 0; i < 10; i++)
        ret[i] /= tot;
    return ret;
}

void print_possible(possible x){
    for(int i=0;i<10;i++)
        cout<<i<<":"<<x[i]<<"  ";
    cout<<endl;
}
#endif
