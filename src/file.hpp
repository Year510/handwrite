#ifndef file_hpp
#define file_hpp

#include "common.hpp"



const int ImgTitleSize = 16;
const int ImgSize = 28 * 28;

const int LabelTitleSize = 8;
const int LabelSize = 1;

inline void inputImg(const string& PATH,int size,int start,image*& pimg){
//先不考虑路径问题，把路径问题向上传播到FilePath.hpp解决
//存储模型ImgTitleSize + [ImgSize,rank0] + [ImgSize,rank1] + ... + [ImgSize,rankN]
//rank i 首字符 ImgTitleSize + ImgSize * i +
    ifstream in=open_read(PATH);
    in.seekg(ImgTitleSize + start * ImgSize );

    uint8_t* buffer = new uint8_t[ImgSize*size];
    in.read(reinterpret_cast<char*>(buffer), ImgSize*size);

    pimg = new image[size];

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < ImgSize; j++) {
            pimg[i][j] = static_cast<double>(buffer[i * ImgSize + j]) / 255.0;
        }
    }
    delete[] buffer;
}

inline void inputLabel(const string& PATH,int size,int start,int*& plabel){
//存储模型LabelTitleSize + [LabelSize,rank0] + [LabelSize,rank1] + ... + [LabelSize,rankN]
    ifstream in=open_read(PATH);
    in.seekg(LabelTitleSize + start * LabelSize );

    uint8_t* buffer = new uint8_t[LabelSize*size];
    in.read(reinterpret_cast<char*>(buffer), LabelSize*size);

    plabel = new int[size];

    for (int i = 0; i < size; i++) {
        plabel[i] = static_cast<int>(buffer[i]);
    }

    delete[] buffer;
}

#endif