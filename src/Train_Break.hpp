#ifndef Train_Break_hpp
#define Train_Break_hpp

#include "common.hpp"

typedef atomic<bool> atomic_bool;

// 按键只会进入当前拥有焦点的那个终端的输入缓冲。
// 别的窗口里按 Q，_kbhit() 看不到，因此不必再判断窗口句柄。
void monitor(atomic_bool &flag, atomic_bool &need_exit){
    while(1){
        if(flag){
            return;
        }

        if(_kbhit()){
            int ch = _getch();
            if(ch == 'q' || ch == 'Q'){
                need_exit=true;
                printf("收到终止要求，等待训练轮次结束后返回\n");
                return;
            }
        }

        Sleep(50);
    }
}


bool CoreTrain(BasicModel *coremodel,int batch){
    //核心训练代码
    //返回true表示训练结束，false表示继续训练
    int trainRet=coremodel->train(batch);
    if(trainRet==0)
        return false;
    else
        return true;
}

void EndTrain(BasicModel *coremodel){
    //训练结束后的收尾工作
    coremodel->output();
    return;
}

int Train_Break_func(BasicModel *coremodel,int batch){
    //还需要记录返回方式，1暂停，0训练完成的结束
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    //程序整体终止有两种可能,均需要子线程先终止，再终止父线程
    //1.父线程检测到终止条件，通知子线程终止
    //2.子线程检测到终止条件，通知父线程终止
    //消息传递通过main_finish_flag和main_exit_flag两个变量进行

    atomic_bool main_finish_flag(false);
    atomic_bool main_exit_flag(false);

    thread check_stop(monitor,ref(main_finish_flag), ref(main_exit_flag));

    int t=0;

    while(1){
        bool backflag=CoreTrain(coremodel,batch);
        t++;
        cout<<"第"<<t<<"batch"<<endl;
        if(backflag){
            //训练结束
            cout<<"训练结束"<<endl;
            break;
        }


        if(main_exit_flag){
            //这是子线程导致的终止
            cout<<"收尾工作中..."<<endl;
            EndTrain(coremodel);
            check_stop.join();
            return 1;
        }
    }
    //这是父线程导致的终止(完成训练)
    //给子线程发送终止信号
    main_finish_flag=true;
    EndTrain(coremodel);
    check_stop.join();
    return 0;
}

#endif