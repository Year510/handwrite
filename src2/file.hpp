#ifndef FILE_HPP
#define FILE_HPP

#include "common.hpp"

#include <vector>

const int ImgTitleSize = 16;
const int ImgSize = Image_Size * Image_Size;
const int LabelTitleSize = 8;
const int LabelSize = 1;
const std::uint32_t kImageMagic = 2051;
const std::uint32_t kLabelMagic = 2049;

inline bool read_u32_be(ifstream& in, std::uint32_t& value) {
    unsigned char b[4];
    in.read(reinterpret_cast<char*>(b), 4);
    if (in.gcount() != 4) return false;
    value = (std::uint32_t(b[0]) << 24) | (std::uint32_t(b[1]) << 16) |
            (std::uint32_t(b[2]) << 8) | std::uint32_t(b[3]);
    return true;
}

inline bool mnist_image_count(const string& path, int& count) {
    ifstream in(path, std::ios::binary);
    if (!in) return false;
    std::uint32_t magic = 0, n = 0, rows = 0, cols = 0;
    if (!read_u32_be(in, magic) || !read_u32_be(in, n) || !read_u32_be(in, rows) || !read_u32_be(in, cols))
        return false;
    if (magic != kImageMagic || rows != 28 || cols != 28 || n == 0) return false;
    count = static_cast<int>(n);
    return true;
}

inline bool mnist_label_count(const string& path, int& count) {
    ifstream in(path, std::ios::binary);
    if (!in) return false;
    std::uint32_t magic = 0, n = 0;
    if (!read_u32_be(in, magic) || !read_u32_be(in, n)) return false;
    if (magic != kLabelMagic || n == 0) return false;
    count = static_cast<int>(n);
    return true;
}

inline bool inputImg(const string& PATH, int size, int start, image*& pimg) {
    pimg = nullptr;
    if (size <= 0 || start < 0 || size > 100000) return false;
    ifstream in(PATH, std::ios::binary);
    if (!in) {
        cerr << "无法打开图像文件: " << PATH << endl;
        return false;
    }
    in.seekg(ImgTitleSize + static_cast<std::streamoff>(start) * ImgSize);
    if (!in) return false;

    std::vector<std::uint8_t> buffer(static_cast<size_t>(ImgSize) * static_cast<size_t>(size));
    in.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
    if (in.gcount() != static_cast<std::streamsize>(buffer.size())) {
        cerr << "图像读取不完整" << endl;
        return false;
    }

    pimg = new image[size];
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < ImgSize; ++j)
            pimg[i][j] = static_cast<double>(buffer[static_cast<size_t>(i) * ImgSize + j]) / 255.0;
    }
    return true;
}

inline bool inputLabel(const string& PATH, int size, int start, int*& plabel) {
    plabel = nullptr;
    if (size <= 0 || start < 0 || size > 100000) return false;
    ifstream in(PATH, std::ios::binary);
    if (!in) {
        cerr << "无法打开标签文件: " << PATH << endl;
        return false;
    }
    in.seekg(LabelTitleSize + static_cast<std::streamoff>(start) * LabelSize);
    if (!in) return false;

    std::vector<std::uint8_t> buffer(static_cast<size_t>(LabelSize) * static_cast<size_t>(size));
    in.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
    if (in.gcount() != static_cast<std::streamsize>(buffer.size())) {
        cerr << "标签读取不完整" << endl;
        return false;
    }

    plabel = new int[size];
    for (int i = 0; i < size; ++i)
        plabel[i] = static_cast<int>(buffer[static_cast<size_t>(i)]);
    return true;
}

inline bool BasicModel::bind_dataset(const string& img, const string& label) {
    int ic = 0, lc = 0;
    if (!mnist_image_count(img, ic)) {
        cerr << "无法读取图像文件头: " << img << endl;
        return false;
    }
    if (!mnist_label_count(label, lc)) {
        cerr << "无法读取标签文件头: " << label << endl;
        return false;
    }
    if (ic != lc) {
        cerr << "图像数量与标签数量不一致" << endl;
        return false;
    }
    return adopt_progress(model_name, finished(), img, label, trained_count(), stored_limit(), ic);
}

inline bool BasicModel::input(ifstream& in) {
    if (!in.is_open()) {
        cerr << "input file open failed" << endl;
        return false;
    }
    std::stringstream ss;
    ss << in.rdbuf();
    const string text = ss.str();

    string file_name, img, label, checksum;
    int img_num = 0, limit = 0, total = 0;
    bool done = false;
    if (!extract_string(text, "model_name", file_name) ||
        !extract_bool(text, "train_finished", done) ||
        !extract_string(text, "train_img_path", img) ||
        !extract_string(text, "train_label_path", label) ||
        !extract_int(text, "train_img_num", img_num) ||
        !extract_int(text, "train_limit", limit) ||
        !extract_int(text, "dataset_total", total) ||
        !extract_string(text, "checksum", checksum)) {
        cerr << "json 缺少公共字段" << endl;
        return false;
    }
    if (!adopt_progress(file_name, done, img, label, img_num, limit, total)) return false;
    if (!read_payload(text)) {
        cerr << "json 参数段无法解析" << endl;
        return false;
    }

    std::uint64_t expect = 0;
    try {
        expect = std::stoull(checksum, nullptr, 16);
    } catch (...) {
        cerr << "json 校验和格式错误" << endl;
        return false;
    }
    if (state_hash() != expect) {
        cerr << "json 校验和不匹配，文件可能被修改" << endl;
        return false;
    }

    int ic = 0, lc = 0;
    if (!mnist_image_count(image_path(), ic) || !mnist_label_count(label_path(), lc)) {
        cerr << "json 中的数据集路径无效" << endl;
        return false;
    }
    if (ic != stored_total() || lc != stored_total()) {
        cerr << "数据集数量与 json 记录不一致" << endl;
        return false;
    }
    if (trained_count() < 0 || trained_count() > stored_total()) return false;
    if (stored_limit() >= 0 && trained_count() > stored_limit()) return false;
    return true;
}

// 一轮训练由模型自己取出下一批 MNIST，再进入派生类的 train。
inline StepResult BasicModel::step(int batch_size) {
    if (batch_size <= 0 || stored_total() < 0 || image_path().empty() || label_path().empty())
        return StepResult::Error;
    const int cap = target_count();
    if (image_cursor() >= cap) {
        mark_progress(image_cursor(), true);
        return StepResult::Finished;
    }
    int n = batch_size;
    const int remain = cap - image_cursor();
    if (n > remain) n = remain;

    const int start = image_cursor();
    image* imgs = nullptr;
    int* labels = nullptr;
    if (!inputImg(image_path(), n, start, imgs)) return StepResult::Error;
    if (!inputLabel(label_path(), n, start, labels)) {
        delete[] imgs;
        return StepResult::Error;
    }
    remember_batch(start, imgs[0]);
    train(imgs, labels, n);
    delete[] imgs;
    delete[] labels;

    const int next = start + n;
    const bool done = next >= cap;
    mark_progress(next, done);
    return done ? StepResult::Finished : StepResult::Continue;
}

#endif
