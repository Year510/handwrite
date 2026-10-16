#ifndef MODEL1_HPP
#define MODEL1_HPP

#include "common.hpp"

#include <charconv>
#include <limits>
#include <random>
#include <system_error>

class Model1 : public BasicModel {
private:
    double AnsWeight[10][784];

    double activ(double x) const {
        double t1 = std::exp(x), t2 = std::exp(-x);
        return (t1 + t2) / 2.0;
    }

    double grad(double c, double x0) const {
        double tmp = std::exp(c * x0);
        double tmp2 = 1.0 / tmp;
        return (tmp - tmp2) / 2.0;
    }

    void write_double(std::ostream& out, double v) const {
        char buf[64];
        auto res = std::to_chars(buf, buf + sizeof buf, v, std::chars_format::general,
                                 std::numeric_limits<double>::max_digits10);
        if (res.ec != std::errc()) out << "0";
        else out.write(buf, res.ptr - buf);
    }

protected:
    void write_payload(std::ostream& out) const override {
        out << "  \"payload\": {\n";
        out << "    \"type\": \"weight\",\n";
        out << "    \"rows\": 10,\n";
        out << "    \"cols\": 784,\n";
        out << "    \"data\": [";
        for (int i = 0; i < 10; ++i) {
            for (int j = 0; j < 784; ++j) {
                if (i || j) out << ", ";
                write_double(out, AnsWeight[i][j]);
            }
        }
        out << "]\n  }";
    }

    bool read_payload(const string& text) override {
        string type;
        int rows = 0, cols = 0;
        if (!extract_string(text, "type", type) || type != "weight") return false;
        if (!extract_int(text, "rows", rows) || !extract_int(text, "cols", cols)) return false;
        if (rows != 10 || cols != 784) return false;

        auto key = text.find("\"data\"");
        if (key == string::npos) return false;
        auto lb = text.find('[', key);
        if (lb == string::npos) return false;
        const char* p = text.c_str() + lb + 1;
        const char* end = text.c_str() + text.size();
        for (int i = 0; i < 10; ++i) {
            for (int j = 0; j < 784; ++j) {
                while (p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t' || *p == ',')) ++p;
                double v = 0.0;
                auto res = std::from_chars(p, end, v);
                if (res.ec != std::errc()) return false;
                AnsWeight[i][j] = v;
                p = res.ptr;
            }
        }
        return true;
    }

    void hash_parameters(Hasher& h) const override {
        h.add_bytes(AnsWeight, sizeof AnsWeight);
    }

public:
    Model1() {
        model_name = "Model1";
        std::mt19937 rng(1);
        std::uniform_real_distribution<double> dist(-0.01, 0.01);
        for (int i = 0; i < 10; ++i)
            for (int j = 0; j < 784; ++j)
                AnsWeight[i][j] = dist(rng);
    }

    possible solve(const image& input_img) override {
        possible basic1;
        for (int i = 0; i < 10; ++i) {
            double solvep = 0.0;
            for (int j = 0; j < 784; ++j) {
                double tmp = AnsWeight[i][j] * input_img[j];
                solvep += activ(tmp);
            }
            basic1[i] = solvep;
        }
        return Normal(basic1);
    }

    void train(const image* front, const int* label, int size) override {
        if (size <= 0) {
            cerr << "train size must be positive!" << endl;
            return;
        }

        double delta[10][784] = {};
        const double alpha = 0.01;

        for (int i = 0; i < size; ++i) {
            possible basic1;
            for (int j = 0; j < 10; ++j) {
                double solvep = 0.0;
                for (int k = 0; k < 784; ++k) {
                    double tmp = AnsWeight[j][k] * front[i][k];
                    solvep += activ(tmp);
                }
                basic1[j] = solvep;
            }

            possible basic2 = Normal(basic1);
            for (int j = 0; j < 10; ++j) {
                double loss = basic2[j] - (j == label[i] ? 1.0 : 0.0);
                for (int k = 0; k < 784; ++k) {
                    double this_color = front[i][k];
                    double wgrad = grad(this_color, AnsWeight[j][k]);
                    delta[j][k] += loss * wgrad * this_color;
                }
            }
        }

        for (int j = 0; j < 10; ++j)
            for (int k = 0; k < 784; ++k)
                AnsWeight[j][k] -= delta[j][k] * alpha / size;
    }

    int weight_mismatch(const Model1& other) const {
        int diff = 0;
        for (int i = 0; i < 10; ++i)
            for (int j = 0; j < 784; ++j)
                if (AnsWeight[i][j] != other.AnsWeight[i][j]) ++diff;
        return diff;
    }
};

#endif
