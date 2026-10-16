#ifndef COMMON_HPP
#define COMMON_HPP

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <conio.h>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <windows.h>

using std::cerr;
using std::cin;
using std::cout;
using std::endl;
using std::function;
using std::ifstream;
using std::ios;
using std::ofstream;
using std::string;

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

enum class StepResult { Continue, Finished, Error };

class Hasher {
    std::uint64_t h = 14695981039346656037ull;
public:
    void add_bytes(const void* data, size_t n) {
        const auto* bytes = static_cast<const unsigned char*>(data);
        for (size_t i = 0; i < n; ++i) {
            h ^= bytes[i];
            h *= 1099511628211ull;
        }
    }
    void add_str(const string& s) {
        add_bytes(s.data(), s.size());
        unsigned char end = 0;
        add_bytes(&end, 1);
    }
    void add_int(int v) { add_bytes(&v, sizeof v); }
    void add_bool(bool v) {
        unsigned char c = v ? 1 : 0;
        add_bytes(&c, 1);
    }
    std::uint64_t value() const { return h; }
};

inline void write_json_string(std::ostream& out, const string& s) {
    out << '"';
    for (unsigned char c : s) {
        if (c == '\\' || c == '"') out << '\\';
        out << static_cast<char>(c);
    }
    out << '"';
}

inline string format_checksum(std::uint64_t sum) {
    std::ostringstream os;
    os << std::hex << std::setfill('0') << std::setw(16) << static_cast<unsigned long long>(sum);
    return os.str();
}

inline size_t find_json_key(const string& text, const string& key) {
    return text.find("\"" + key + "\"");
}

inline bool json_value_at(const string& text, const string& key, size_t& pos) {
    size_t k = find_json_key(text, key);
    if (k == string::npos) return false;
    size_t colon = text.find(':', k + key.size() + 2);
    if (colon == string::npos) return false;
    pos = colon + 1;
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) ++pos;
    return pos < text.size();
}

inline bool extract_string(const string& text, const string& key, string& out) {
    size_t pos = 0;
    if (!json_value_at(text, key, pos) || text[pos] != '"') return false;
    string val;
    for (size_t i = pos + 1; i < text.size(); ++i) {
        if (text[i] == '\\') {
            if (i + 1 >= text.size()) return false;
            char n = text[++i];
            val.push_back(n == 'n' ? '\n' : n);
            continue;
        }
        if (text[i] == '"') {
            out = val;
            return true;
        }
        val.push_back(text[i]);
    }
    return false;
}

inline bool extract_bool(const string& text, const string& key, bool& out) {
    size_t pos = 0;
    if (!json_value_at(text, key, pos)) return false;
    if (text.compare(pos, 4, "true") == 0) {
        out = true;
        return true;
    }
    if (text.compare(pos, 5, "false") == 0) {
        out = false;
        return true;
    }
    return false;
}

inline bool extract_int(const string& text, const string& key, int& out) {
    size_t pos = 0;
    if (!json_value_at(text, key, pos)) return false;
    try {
        size_t used = 0;
        int v = std::stoi(text.substr(pos), &used);
        if (used == 0) return false;
        out = v;
        return true;
    } catch (...) {
        return false;
    }
}

inline possible Normal(const possible& x) {
    double m = x[0];
    for (int i = 1; i < 10; ++i)
        if (x[i] > m) m = x[i];
    double tot = 0.0;
    possible ret;
    for (int i = 0; i < 10; ++i) {
        ret[i] = std::exp(x[i] - m);
        tot += ret[i];
    }
    for (int i = 0; i < 10; ++i)
        ret[i] /= tot;
    return ret;
}

inline void print_possible(possible x) {
    for (int i = 0; i < 10; ++i)
        cout << i << ":" << x[i] << "  ";
    cout << endl;
}

// 继续训练时还不知道派生类，所以模型名在构造对象之前单独读出来。
inline string getJsonModelName(const string& path) {
    ifstream in(path, std::ios::binary);
    if (!in) return "";
    std::stringstream ss;
    ss << in.rdbuf();
    string name;
    if (!extract_string(ss.str(), "model_name", name)) return "";
    return name;
}

// 公共 JSON 记模型名、路径、进度和校验和。参数段由派生类写。
// 校验和覆盖这些字段和参数本体，用来拒绝被改过的文件。
class BasicModel {
protected:
    string model_name;
    virtual void write_payload(std::ostream& out) const = 0;
    virtual bool read_payload(const string& text) = 0;
    virtual void hash_parameters(Hasher& h) const = 0;

private:
    bool train_finished;
    string train_img_path;
    string train_label_path;
    string save_path_;
    int train_img_num;
    int train_limit;
    int dataset_total;
    int last_load_start_;
    double last_image_sum_;

public:
    BasicModel()
        : train_finished(false),
          train_img_num(0),
          train_limit(-1),
          dataset_total(-1),
          last_load_start_(-1),
          last_image_sum_(0) {}
    virtual ~BasicModel() = default;

    virtual possible solve(const image& input_img) = 0;
    virtual void train(const image* front, const int* label, int size) = 0;

    bool input(ifstream& in);
    bool bind_dataset(const string& img, const string& label);
    StepResult step(int batch_size);

    bool output(ofstream& out) const {
        if (!out.is_open()) {
            cerr << "output file open failed" << endl;
            return false;
        }
        out << "{\n";
        out << "  \"model_name\": ";
        write_json_string(out, model_name);
        out << ",\n";
        out << "  \"train_finished\": " << (train_finished ? "true" : "false") << ",\n";
        out << "  \"train_img_path\": ";
        write_json_string(out, train_img_path);
        out << ",\n";
        out << "  \"train_label_path\": ";
        write_json_string(out, train_label_path);
        out << ",\n";
        out << "  \"train_img_num\": " << train_img_num << ",\n";
        out << "  \"train_limit\": " << train_limit << ",\n";
        out << "  \"dataset_total\": " << dataset_total << ",\n";
        out << "  \"checksum\": \"" << format_checksum(state_hash()) << "\",\n";
        write_payload(out);
        out << "\n}\n";
        return static_cast<bool>(out);
    }

    std::uint64_t state_hash() const {
        Hasher h;
        h.add_str(model_name);
        h.add_bool(train_finished);
        h.add_str(train_img_path);
        h.add_str(train_label_path);
        h.add_int(train_img_num);
        h.add_int(train_limit);
        h.add_int(dataset_total);
        hash_parameters(h);
        return h.value();
    }

    void set_train_limit(int n) { train_limit = n; }
    void set_save_path(string path) { save_path_ = path; }

    const string& name() const { return model_name; }
    int trained_count() const { return train_img_num; }
    bool finished() const { return train_finished; }
    int last_load_start() const { return last_load_start_; }
    double last_image_sum() const { return last_image_sum_; }

    int target_count() const {
        if (dataset_total < 0) return train_limit < 0 ? 0 : train_limit;
        if (train_limit < 0) return dataset_total;
        return train_limit < dataset_total ? train_limit : dataset_total;
    }

    string save_path() const {
        if (!save_path_.empty()) return save_path_;
        return string("Model_Matrix/") + model_name + ".json";
    }

protected:
    void remember_batch(int start, const image& first) {
        last_load_start_ = start;
        double sum = 0.0;
        for (int i = 0; i < Image_Size * Image_Size; ++i) sum += first[i];
        last_image_sum_ = sum;
    }

    void mark_progress(int new_count, bool done) {
        train_img_num = new_count;
        train_finished = done;
    }

    int image_cursor() const { return train_img_num; }
    int stored_total() const { return dataset_total; }
    int stored_limit() const { return train_limit; }
    const string& image_path() const { return train_img_path; }
    const string& label_path() const { return train_label_path; }

    bool adopt_progress(const string& file_name, bool done, const string& img, const string& label,
                        int img_num, int limit, int total) {
        if (file_name != model_name) {
            cerr << "json 模型名与对象不符" << endl;
            return false;
        }
        train_finished = done;
        train_img_path = img;
        train_label_path = label;
        train_img_num = img_num;
        train_limit = limit;
        dataset_total = total;
        return true;
    }
};

#endif
