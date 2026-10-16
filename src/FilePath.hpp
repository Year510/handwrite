#ifndef FILEPATH_HPP
#define FILEPATH_HPP

#include <filesystem>
#include <fstream>
#include <string>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

using namespace std;
using namespace std::filesystem;

// 从 dir 向上找 handwrite 根目录。标志是 MNIST_DATA/train-images-idx3-ubyte 存在。
inline path search_up(path dir) {
    for (int i = 0; i < 16 && !dir.empty(); ++i) {
        if (is_regular_file(dir / "MNIST_DATA" / "train-images-idx3-ubyte"))
            return dir;
        path parent = dir.parent_path();
        if (parent == dir) break;
        dir = parent;
    }
    return {};
}

inline path exe_dir() {
    wchar_t buf[MAX_PATH];
    DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    if (n == 0 || n >= MAX_PATH) return {};
    return path(buf).parent_path();
}

// 先看当前目录，再看 exe 旁边，最后看本头文件编译时所在的位置。
inline path project_root() {
    static path cached = [] {
        path found = search_up(current_path());
        if (found.empty()) found = search_up(exe_dir());
        path self(__FILE__);
        if (found.empty() && self.is_absolute())
            found = search_up(self.parent_path());
        return found;
    }();
    return cached;
}

// 相对 handwrite 的路径变成绝对路径。找不到根目录时返回空 path。
inline path resolve(const string& rel) {
    path p(rel);
    if (p.empty()) return {};
    if (!p.is_absolute()) {
        path root = project_root();
        if (root.empty()) return {};
        p = root / p;
    }
    p = absolute(p);
    p.make_preferred();
    return p;
}

// 读写都带 binary。文本模式在 Windows 上会把 \n 写成 \r\n，
// 再用二进制读回来时，getline 的行尾会多一个 \r。MNIST 也必须按二进制读。
inline ifstream open_read(const string& rel) {
    return ifstream(resolve(rel), ios::in | ios::binary);
}

inline ofstream open_write(const string& rel) {
    path file = resolve(rel);
    if (!file.empty() && file.has_parent_path())
        create_directories(file.parent_path());
    return ofstream(file, ios::out | ios::binary);
}

#endif
