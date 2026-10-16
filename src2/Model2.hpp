#ifndef MODEL2_HPP
#define MODEL2_HPP

#include "common.hpp"

// 不学习的对照模型。solve 固定返回 10 个 0.1，用来确认同一套调度能走到另一个派生类。
class Model2 : public BasicModel {
protected:
    void write_payload(std::ostream& out) const override {
        out << "  \"payload\": {\"type\": \"uniform\", \"value\": 0.1}";
    }

    bool read_payload(const string& text) override {
        string type;
        if (!extract_string(text, "type", type)) return false;
        return type == "uniform";
    }

    void hash_parameters(Hasher& h) const override {
        h.add_str("uniform");
        double value = 0.1;
        h.add_bytes(&value, sizeof value);
    }

public:
    Model2() { model_name = "Model2"; }

    possible solve(const image&) override {
        possible p;
        for (int i = 0; i < 10; ++i) p[i] = 0.1;
        return p;
    }

    void train(const image*, const int*, int) override {}
};

#endif
