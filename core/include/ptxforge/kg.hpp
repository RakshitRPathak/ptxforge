// kg.hpp — symptom diagnosis + knowledge-graph retrieval ("GraphRAG-lite").
//
// Graph:  symptom --remedy(param, action)--> parameter move --side_effects--> other symptoms.
// Retrieval: for each symptom present in the profile, pull its remedy edges, walk one hop along
// `side_effects`, and rank remedies by (symptom weight) - (penalty for NEW side-effect symptoms)
// - (penalty for remedies that worsen a heavier symptom that is already present).
// The same retrieved subgraph is rendered as text for the external (LLM) advisor.
#pragma once

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "ptxforge/cost_model.hpp"
#include "ptxforge/json.hpp"

namespace ptxforge {

struct Remedy {
  std::string id, symptom, param, action, why;
  std::vector<std::string> side_effects;
  std::string evidence;
  double score = 0;
};

struct KnowledgeGraph {
  std::map<std::string, double> weights;
  std::vector<Remedy> edges;

  static KnowledgeGraph from_json(const json::Value &v) {
    KnowledgeGraph kg;
    const auto &sy = v.at("symptoms");
    for (size_t i = 0; i < sy.keys().size(); ++i)
      kg.weights[sy.keys()[i]] = sy.values()[i].at("weight").as_number();
    for (const auto &r : v.at("remedies").items()) {
      Remedy e;
      e.id = r.at("id").as_string();
      e.symptom = r.at("symptom").as_string();
      e.param = r.at("param").as_string();
      e.action = r.at("action").as_string();
      e.why = r.at("why").as_string();
      for (const auto &s : r.at("side_effects").items()) e.side_effects.push_back(s.as_string());
      if (const auto *ev = r.find("evidence")) e.evidence = ev->as_string();
      kg.edges.push_back(std::move(e));
    }
    return kg;
  }

  static KnowledgeGraph load(const std::string &path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open knowledge graph: " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return from_json(json::Value::parse(ss.str()));
  }

  double weight(const std::string &s, double fallback) const {
    auto it = weights.find(s);
    return it == weights.end() ? fallback : it->second;
  }
};

// Symptoms from MEASURED profile fields where available, modeled ones otherwise.
inline std::vector<std::string> diagnose(const Config &c, const Analysis &an, const Profile &prof) {
  std::vector<std::string> s;
  const bool spill = prof.local_bytes ? *prof.local_bytes > 0 : an.spill_regs > 0;
  if (spill) s.push_back("register_spill");
  if (prof.occupancy.value_or(an.occupancy) < 0.25) s.push_back("low_occupancy");
  if (an.bank_degree >= 2) s.push_back("bank_conflict");
  if (an.memory_bound) s.push_back("memory_bound");
  if (c.TM * c.TN < 16) s.push_back("low_ilp");
  if (an.waves < 1.0) s.push_back("underfilled_gpu");
  return s;
}

inline std::vector<Remedy> retrieve(const std::vector<std::string> &symptoms, const KnowledgeGraph &kg) {
  const std::set<std::string> present(symptoms.begin(), symptoms.end());
  std::vector<Remedy> out;
  for (const Remedy &r : kg.edges) {
    if (!present.count(r.symptom)) continue;
    double penalty = 0, conflict = 0;
    for (const auto &se : r.side_effects) {
      if (!present.count(se)) penalty += 0.5 * kg.weight(se, 0.3);               // new risk
      else if (kg.weight(se, 0.0) > kg.weight(r.symptom, 0.0)) conflict += 1.0;  // worsens a heavier symptom
    }
    Remedy e = r;
    e.score = std::round((kg.weight(r.symptom, 0.0) - penalty - conflict) * 1000.0) / 1000.0;
    out.push_back(std::move(e));
  }
  std::stable_sort(out.begin(), out.end(), [](const Remedy &a, const Remedy &b) { return a.score > b.score; });
  return out;
}

inline std::string subgraph_text(const std::vector<std::string> &symptoms, const std::vector<Remedy> &rem) {
  std::string t = "Observed symptoms: ";
  for (size_t i = 0; i < symptoms.size(); ++i) t += (i ? ", " : "") + symptoms[i];
  if (symptoms.empty()) t += "none";
  t += "\nCandidate remedies (best first):";
  for (size_t i = 0; i < rem.size() && i < 8; ++i) {
    const Remedy &r = rem[i];
    std::string risks;
    for (size_t k = 0; k < r.side_effects.size(); ++k) risks += (k ? ", " : "") + r.side_effects[k];
    if (risks.empty()) risks = "none";
    char score[32];
    std::snprintf(score, sizeof score, "%g", r.score);
    t += "\n- [" + r.id + "] " + r.symptom + ": " + r.action + " " + r.param + " (score " + score + "). " +
         r.why + " Risks: " + risks + ". Source family: " + r.evidence + ".";
  }
  return t;
}

} // namespace ptxforge
