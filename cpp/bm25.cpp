// Okapi BM25 inverted index, exposed to Python as conduit._bm25.
//
// Keyword search is the hot path of hybrid retrieval: every query walks the
// postings lists of its terms. Doing that in C++ keeps it fast as the corpus
// grows, while tokenization stays in Python so both backends see identical
// tokens. conduit/search/bm25.py holds a pure-Python twin with the same maths;
// tests/test_bm25.py checks the two agree.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace py = pybind11;

class BM25Index {
public:
    explicit BM25Index(double k1 = 1.5, double b = 0.75) : k1_(k1), b_(b) {}

    void add(int doc_id, const std::vector<std::string>& tokens) {
        if (slot_of_.count(doc_id)) {
            throw std::invalid_argument("duplicate doc id " + std::to_string(doc_id));
        }
        const size_t slot = doc_len_.size();
        slot_of_[doc_id] = slot;
        doc_ids_.push_back(doc_id);
        doc_len_.push_back(static_cast<double>(tokens.size()));
        total_len_ += static_cast<double>(tokens.size());

        std::unordered_map<std::string, int> tf;
        for (const auto& t : tokens) tf[t]++;
        for (const auto& [term, freq] : tf) postings_[term].push_back({slot, freq});
    }

    // Scores only the documents in `allowed`. Filtering happens before ranking,
    // so a document the caller may not see can never occupy a top-k slot.
    std::vector<std::pair<int, double>> search(const std::vector<std::string>& query,
                                               size_t k,
                                               const std::vector<int>& allowed) const {
        std::vector<std::pair<int, double>> out;
        if (doc_len_.empty() || k == 0) return out;

        std::vector<char> mask(doc_len_.size(), 0);
        for (int id : allowed) {
            auto it = slot_of_.find(id);
            if (it != slot_of_.end()) mask[it->second] = 1;
        }

        const double n = static_cast<double>(doc_len_.size());
        const double avgdl = total_len_ > 0 ? total_len_ / n : 1.0;
        std::vector<double> scores(doc_len_.size(), 0.0);
        std::vector<char> hit(doc_len_.size(), 0);
        std::unordered_set<std::string> seen;

        for (const auto& term : query) {
            if (!seen.insert(term).second) continue;  // repeated query terms count once
            auto it = postings_.find(term);
            if (it == postings_.end()) continue;
            const double df = static_cast<double>(it->second.size());
            const double idf = std::log(1.0 + (n - df + 0.5) / (df + 0.5));
            for (const auto& p : it->second) {
                if (!mask[p.slot]) continue;
                const double f = static_cast<double>(p.tf);
                const double norm = k1_ * (1.0 - b_ + b_ * doc_len_[p.slot] / avgdl);
                scores[p.slot] += idf * f * (k1_ + 1.0) / (f + norm);
                hit[p.slot] = 1;
            }
        }

        for (size_t s = 0; s < scores.size(); ++s) {
            if (hit[s]) out.emplace_back(doc_ids_[s], scores[s]);
        }
        auto by_score = [](const auto& a, const auto& b) {
            return a.second != b.second ? a.second > b.second : a.first < b.first;
        };
        if (out.size() > k) {
            std::partial_sort(out.begin(), out.begin() + static_cast<long>(k), out.end(), by_score);
            out.resize(k);
        } else {
            std::sort(out.begin(), out.end(), by_score);
        }
        return out;
    }

    size_t size() const { return doc_len_.size(); }

private:
    struct Posting {
        size_t slot;
        int tf;
    };

    double k1_;
    double b_;
    double total_len_ = 0.0;
    std::vector<int> doc_ids_;
    std::vector<double> doc_len_;
    std::unordered_map<int, size_t> slot_of_;
    std::unordered_map<std::string, std::vector<Posting>> postings_;
};

PYBIND11_MODULE(_bm25, m) {
    m.doc() = "BM25 inverted index (C++)";
    py::class_<BM25Index>(m, "BM25Index")
        .def(py::init<double, double>(), py::arg("k1") = 1.5, py::arg("b") = 0.75)
        .def("add", &BM25Index::add, py::arg("doc_id"), py::arg("tokens"))
        .def("search", &BM25Index::search, py::arg("query"), py::arg("k"), py::arg("allowed"),
             py::call_guard<py::gil_scoped_release>())
        .def("__len__", &BM25Index::size);
}
