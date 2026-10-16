#ifndef TRAIN_BREAK_HPP
#define TRAIN_BREAK_HPP

#include "file.hpp"

#include <filesystem>
#include <stdexcept>

typedef std::atomic<bool> atomic_bool;

struct TrainConfig {
    int batch_size = 50;
    int pause_after_images = -1;
};

struct TrainSession {
    BasicModel* model = nullptr;
    TrainConfig config{};
    atomic_bool* exit_flag = nullptr;
    string error;
};

// 按键只会进入当前拥有焦点的那个终端的输入缓冲。
// 别的窗口里按 Q，_kbhit() 看不到，因此不必再判断窗口句柄。
inline void monitor(atomic_bool& flag, atomic_bool& need_exit) {
    while (true) {
        if (flag.load()) return;
        if (_kbhit()) {
            int ch = _getch();
            if (ch == 'q' || ch == 'Q') {
                need_exit.store(true);
                std::printf("收到终止要求，等待训练轮次结束后返回\n");
                return;
            }
        }
        Sleep(50);
    }
}

inline bool CoreTrain(TrainSession& session) {
    BasicModel* model = session.model;
    if (!model || !session.exit_flag) {
        session.error = "训练对象无效";
        return false;
    }
    if (session.config.pause_after_images >= 0 &&
        model->trained_count() >= session.config.pause_after_images) {
        session.exit_flag->store(true);
        return false;
    }

    StepResult result = model->step(session.config.batch_size);
    if (result == StepResult::Error) {
        session.error = "读取 MNIST 或训练步进失败";
        session.exit_flag->store(true);
        return false;
    }
    cout << "已训练 " << model->trained_count() << " / " << model->target_count() << endl;
    return result == StepResult::Finished;
}

inline bool EndTrain(BasicModel* model) {
    if (!model) return false;
    try {
        std::filesystem::path path(model->save_path());
        if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
        ofstream out(path);
        if (!model->output(out)) return false;
        out.flush();
        if (!out.good()) return false;
        cout << "已写入 " << path.generic_string() << endl;
        return true;
    } catch (const std::exception& ex) {
        cerr << "保存失败: " << ex.what() << endl;
        return false;
    }
}

// 返回 0 表示训练完成，1 表示暂停，2 表示出错。完成和暂停都会把 json 写到 Model_Matrix。
inline int Train_Break_func(BasicModel* coremodel, TrainConfig config = {}) {
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    if (!coremodel) return 2;

    atomic_bool main_finish_flag(false);
    atomic_bool main_exit_flag(false);
    TrainSession session;
    session.model = coremodel;
    session.config = config;
    session.exit_flag = &main_exit_flag;

    std::thread check_stop(monitor, std::ref(main_finish_flag), std::ref(main_exit_flag));

    int status = 0;
    while (true) {
        bool done = CoreTrain(session);
        if (!session.error.empty()) {
            status = 2;
            break;
        }
        if (done) {
            cout << "训练结束" << endl;
            status = EndTrain(coremodel) ? 0 : 2;
            break;
        }
        if (main_exit_flag.load()) {
            cout << "收尾工作中..." << endl;
            status = EndTrain(coremodel) ? 1 : 2;
            break;
        }
    }

    main_finish_flag.store(true);
    check_stop.join();
    return status;
}

#endif
