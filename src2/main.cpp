#include "common.hpp"
#include "file.hpp"
#include "Model1.hpp"
#include "Model2.hpp"
#include "Train_Break.hpp"

#include <filesystem>

const char* kTrainImgName = "MNIST_DATA/train-images-idx3-ubyte";
const char* kTrainLabelName = "MNIST_DATA/train-labels-idx1-ubyte";
constexpr int kBatch = 50;
constexpr int kTotal = 200;
constexpr int kPause = 100;
constexpr int kMid = 150;

string g_root;

string join_root(const string& rel) { return g_root + rel; }

void locate_root() {
    namespace fs = std::filesystem;
    if (fs::exists(kTrainImgName)) g_root = "";
    else if (fs::exists(string("../") + kTrainImgName)) g_root = "../";
    else g_root = "";
}

BasicModel* create_model(const string& name) {
    if (name == "Model1") return new Model1();
    if (name == "Model2") return new Model2();
    return nullptr;
}

BasicModel* load_saved_model(const string& path) {
    string name = getJsonModelName(path);
    if (name.empty()) {
        cerr << "无法从 json 识别模型" << endl;
        return nullptr;
    }
    BasicModel* model = create_model(name);
    if (!model) {
        cerr << "未知模型: " << name << endl;
        return nullptr;
    }
    ifstream in(path, std::ios::binary);
    if (!model->input(in)) {
        delete model;
        return nullptr;
    }
    return model;
}

int run_session(BasicModel* model, int pause_after) {
    TrainConfig cfg;
    cfg.batch_size = kBatch;
    cfg.pause_after_images = pause_after;
    return Train_Break_func(model, cfg);
}

double image_sum_at(int index) {
    image* p = nullptr;
    if (!inputImg(join_root(kTrainImgName), 1, index, p) || !p) return -1.0;
    double sum = 0.0;
    for (int i = 0; i < Image_Size * Image_Size; ++i) sum += (*p)[i];
    delete[] p;
    return sum;
}

bool is_uniform(const possible& p) {
    for (int i = 0; i < 10; ++i)
        if (p[i] != 0.1) return false;
    return true;
}

bool corrupt_first_weight(const string& src, const string& dst) {
    ifstream in(src, std::ios::binary);
    if (!in) return false;
    std::stringstream ss;
    ss << in.rdbuf();
    string text = ss.str();
    auto key = text.find("\"data\"");
    if (key == string::npos) return false;
    auto num = text.find_first_of("0123456789", key);
    if (num == string::npos) return false;
    text[num] = (text[num] == '9') ? '8' : static_cast<char>(text[num] + 1);
    ofstream out(dst, std::ios::binary);
    out << text;
    return static_cast<bool>(out);
}

void run_pause_reload(const function<void(bool, const string&)>& check, BasicModel* model, const string& tag,
                      const string& pause_file, const string& mid_file, const string& done_file,
                      bool expect_uniform, const image& sample, BasicModel*& finished_out) {
    finished_out = nullptr;
    model->set_save_path(pause_file);
    int rc = run_session(model, kPause);
    check(rc == 1, tag + " 训练到 100 张后暂停");
    check(model->trained_count() == kPause, tag + " 暂停时进度是 100");
    check(!model->finished(), tag + " 暂停时未标记完成");
    check(std::filesystem::exists(pause_file), tag + " 暂停 json 已写入 Model_Matrix");
    check(getJsonModelName(pause_file) == model->name(), tag + " 暂停 json 中的模型名正确");

    if (expect_uniform)
        check(is_uniform(model->solve(sample)), tag + " 暂停时 solve 为均匀分布");

    BasicModel* loaded = load_saved_model(pause_file);
    check(loaded != nullptr, tag + " 重新读入暂停 json");
    if (!loaded) return;
    if (loaded->trained_count() != kPause || loaded->target_count() != kTotal) {
        check(false, tag + " 读回的进度或上限不正确，停止该模型");
        delete loaded;
        return;
    }
    check(loaded->name() == model->name(), tag + " 读回对象的类型与 json 一致");
    check(loaded->state_hash() == model->state_hash(), tag + " 读回后校验和与暂停时一致");

    Model1* paused1 = dynamic_cast<Model1*>(model);
    Model1* loaded1 = dynamic_cast<Model1*>(loaded);
    if (paused1 || loaded1) {
        int diff = (paused1 && loaded1) ? loaded1->weight_mismatch(*paused1) : -1;
        check(diff == 0, tag + " 7840 个权重完全一致，差异 " + std::to_string(diff));
    }

    loaded->set_save_path(mid_file);
    rc = run_session(loaded, kMid);
    check(rc == 1, tag + " 从 100 继续，到 150 再次暂停");
    check(loaded->trained_count() == kMid, tag + " 第二次暂停时进度是 150");
    check(loaded->last_load_start() == kPause, tag + " 续训的第一批从第 100 张开始");
    check(loaded->last_image_sum() == image_sum_at(kPause), tag + " 续训读到的就是第 100 张图");

    BasicModel* mid = load_saved_model(mid_file);
    check(mid && mid->trained_count() == kMid, tag + " 中途 json 可以再读入且进度为 150");
    delete mid;

    loaded->set_save_path(done_file);
    rc = run_session(loaded, -1);
    check(rc == 0, tag + " 继续训练到 200 张并正常结束");
    check(loaded->trained_count() == kTotal && loaded->finished(), tag + " 结束时进度是 200 且已完成");
    check(loaded->last_load_start() == kMid, tag + " 最后一批从第 150 张开始");
    check(loaded->last_image_sum() == image_sum_at(kMid), tag + " 最后一批读到的就是第 150 张图");
    check(loaded->state_hash() != model->state_hash(), tag + " 进度变化后校验和改变");

    if (paused1 && loaded1) {
        int diff = loaded1->weight_mismatch(*paused1);
        check(diff > 0, tag + " 继续训练后权重发生变化，差异 " + std::to_string(diff));
    }
    if (expect_uniform)
        check(is_uniform(loaded->solve(sample)), tag + " 训完后 solve 仍是均匀分布");

    BasicModel* done = load_saved_model(done_file);
    check(done && done->finished() && done->trained_count() == kTotal, tag + " 完成 json 可以读入且进度为 200");
    delete done;
    finished_out = loaded;
}

int probe_keyboard_interrupt() {
    HANDLE h = GetStdHandle(STD_INPUT_HANDLE);
    DWORD mode = 0;
    if (h == INVALID_HANDLE_VALUE || h == nullptr || !GetConsoleMode(h, &mode)) return 0;

    atomic_bool stop(false), need_exit(false);
    std::thread worker(monitor, std::ref(stop), std::ref(need_exit));
    INPUT_RECORD rec{};
    rec.EventType = KEY_EVENT;
    rec.Event.KeyEvent.bKeyDown = TRUE;
    rec.Event.KeyEvent.wRepeatCount = 1;
    rec.Event.KeyEvent.wVirtualKeyCode = 'Q';
    rec.Event.KeyEvent.uChar.AsciiChar = 'q';
    DWORD written = 0;
    BOOL ok = WriteConsoleInputA(h, &rec, 1, &written);
    bool seen = false;
    for (int i = 0; i < 30 && !(seen = need_exit.load()); ++i) Sleep(50);
    stop.store(true);
    worker.join();
    FlushConsoleInputBuffer(h);
    if (!ok || written != 1) return 0;
    return seen ? 1 : -1;
}

int RunFeasibilityTest() {
    locate_root();
    int passes = 0, fails = 0;
    function<void(bool, const string&)> check = [&](bool ok, const string& msg) {
        cout << (ok ? "[PASS] " : "[FAIL] ") << msg << endl;
        if (ok) ++passes;
        else ++fails;
    };

    cout << "可行性测试：两个模型各训练 " << kTotal << " 张，在 " << kPause
         << " 张暂停并写 json，读回后继续" << endl;
    cout << "图像 " << join_root(kTrainImgName) << endl;

    HANDLE hin = GetStdHandle(STD_INPUT_HANDLE);
    if (hin != INVALID_HANDLE_VALUE && hin != nullptr) FlushConsoleInputBuffer(hin);
    int kb = probe_keyboard_interrupt();
    if (kb == 0) cout << "[SKIP] 当前进程没有控制台输入缓冲，Q 键未注入；中断由调度在轮次之间触发" << endl;
    else check(kb == 1, "向控制台注入 Q 后，监察线程置位中断");

    image* sample = nullptr;
    int* sample_label = nullptr;
    bool sample_ok = inputImg(join_root(kTrainImgName), 1, 0, sample) &&
                     inputLabel(join_root(kTrainLabelName), 1, 0, sample_label);
    check(sample_ok, "读出第 0 张图和标签供 solve 使用");
    if (!sample_ok) {
        cout << "RESULT pass=" << passes << " fail=" << fails << endl;
        return 1;
    }

    BasicModel* m1 = new Model1();
    BasicModel* m2 = new Model2();
    m1->set_train_limit(kTotal);
    m2->set_train_limit(kTotal);
    check(m1->bind_dataset(join_root(kTrainImgName), join_root(kTrainLabelName)), "Model1 绑定 MNIST 训练集");
    check(m2->bind_dataset(join_root(kTrainImgName), join_root(kTrainLabelName)), "Model2 绑定 MNIST 训练集");

    BasicModel* m1_done = nullptr;
    BasicModel* m2_done = nullptr;
    if (m1->target_count() == kTotal)
        run_pause_reload(check, m1, "Model1", join_root("Model_Matrix/Model1_pause.json"),
                         join_root("Model_Matrix/Model1_mid.json"), join_root("Model_Matrix/Model1_done.json"),
                         false, *sample, m1_done);
    else
        check(false, "Model1 训练上限不是 200");

    if (m2->target_count() == kTotal)
        run_pause_reload(check, m2, "Model2", join_root("Model_Matrix/Model2_pause.json"),
                         join_root("Model_Matrix/Model2_mid.json"), join_root("Model_Matrix/Model2_done.json"),
                         true, *sample, m2_done);
    else
        check(false, "Model2 训练上限不是 200");

    if (m1_done && m2_done) {
        BasicModel* poly[2] = {m1_done, m2_done};
        possible a = poly[0]->solve(*sample);
        possible b = poly[1]->solve(*sample);
        cout << "样本标签 " << sample_label[0] << endl;
        cout << "Model1 ";
        print_possible(a);
        cout << "Model2 ";
        print_possible(b);
        double sum = 0.0;
        bool differ = false;
        for (int i = 0; i < 10; ++i) {
            sum += a[i];
            if (std::fabs(a[i] - b[i]) > 1e-8) differ = true;
        }
        check(poly[0]->name() == "Model1" && poly[1]->name() == "Model2", "BasicModel* 分别落到 Model1 和 Model2");
        check(std::fabs(sum - 1.0) < 1e-9, "Model1::solve 通过基类指针返回的概率和为 1");
        check(is_uniform(b), "Model2::solve 通过基类指针返回均匀分布");
        check(differ, "同一张图上两个派生类的 solve 结果不同");
    } else {
        check(false, "缺少完成态模型，无法比较多态输出");
    }

    string bad = join_root("Model_Matrix/Model1_bad.json");
    check(corrupt_first_weight(join_root("Model_Matrix/Model1_pause.json"), bad), "生成被改过的 json");
    check(load_saved_model(bad) == nullptr, "篡改权重后重新读入被拒绝");

    Model2 wrong;
    ifstream wrong_in(join_root("Model_Matrix/Model1_pause.json"), std::ios::binary);
    check(!wrong.input(wrong_in), "Model2 拒绝读入 Model1 的 json");
    check(getJsonModelName(join_root("Model_Matrix/no_such_model.json")).empty(), "不存在的 json 返回空模型名");

    delete[] sample;
    delete[] sample_label;
    delete m1;
    delete m2;
    delete m1_done;
    delete m2_done;

    cout << "RESULT pass=" << passes << " fail=" << fails << endl;
    return fails == 0 ? 0 : 1;
}

void StartTrain() {
    locate_root();
    cout << "输入模型号和训练集号（模型 1=Model1 2=Model2，训练集 0=MNIST 训练集）" << endl;
    int model_id = 0, set_id = 0;
    if (!(cin >> model_id >> set_id)) return;

    BasicModel* model = nullptr;
    if (model_id == 1) model = new Model1();
    else if (model_id == 2) model = new Model2();
    else {
        cout << "无效模型号" << endl;
        return;
    }
    if (set_id != 0) {
        cout << "无效训练集号" << endl;
        delete model;
        return;
    }
    if (!model->bind_dataset(join_root(kTrainImgName), join_root(kTrainLabelName))) {
        cout << "数据集打开失败" << endl;
        delete model;
        return;
    }

    cout << "新模型训练开始，共 " << model->target_count() << " 张，按 Q 暂停" << endl;
    int backflag = Train_Break_func(model);
    if (backflag == 0) cout << "模型完成训练" << endl;
    else if (backflag == 1) cout << "已经暂停" << endl;
    else cout << "训练失败" << endl;
    cout << "核心参数 json 路径: " << model->save_path() << endl;
    delete model;
}

void ContinueTrain() {
    cout << "输入模型参数 json 文件地址" << endl;
    string path;
    if (!(cin >> path)) return;
    BasicModel* model = load_saved_model(path);
    if (!model) {
        cout << "json 校验失败或已损坏" << endl;
        return;
    }
    cout << "继续训练 " << model->name() << "，已完成 " << model->trained_count()
         << " 张，按 Q 暂停" << endl;
    int backflag = Train_Break_func(model);
    if (backflag == 0) cout << "模型完成训练" << endl;
    else if (backflag == 1) cout << "已经暂停" << endl;
    else cout << "训练失败" << endl;
    cout << "核心参数 json 路径: " << model->save_path() << endl;
    delete model;
}

void RunFeasibilityMenu() {
    int rc = RunFeasibilityTest();
    cout << "测试返回值 " << rc << endl;
}

function<void()> menu[10];

int main(int argc, char** argv) {
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    if (argc >= 2 && string(argv[1]) == "test") return RunFeasibilityTest();

    menu[1] = StartTrain;
    menu[2] = ContinueTrain;
    menu[3] = RunFeasibilityMenu;

    cout << "请选择操作：1.开始新训练 2.继续训练 3.可行性测试" << endl;
    int op = 0;
    if (!(cin >> op)) {
        cout << "无效操作" << endl;
        return 1;
    }
    if (op >= 1 && op <= 3 && menu[op]) menu[op]();
    else cout << "无效操作" << endl;
    return 0;
}
