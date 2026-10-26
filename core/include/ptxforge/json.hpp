// json.hpp — minimal, dependency-free JSON value/parser/serializer (insertion-ordered objects).
// Enough for traces, the knowledge graph, and the LLM bridge. Not a general-purpose library:
// no comments, no NaN/Inf (serialized as null), numbers are doubles.
#pragma once

#include <cctype>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ptxforge::json {

class Value {
public:
  enum class Type { Null, Bool, Number, String, Array, Object };

  Value() = default;
  Value(std::nullptr_t) {}
  Value(bool b) : type_(Type::Bool), b_(b) {}
  template <std::integral T>
    requires(!std::same_as<T, bool>)
  Value(T i) : type_(Type::Number), n_(static_cast<double>(i)) {}
  Value(double d) : type_(Type::Number), n_(d) {}
  Value(const char *s) : type_(Type::String), s_(s) {}
  Value(std::string s) : type_(Type::String), s_(std::move(s)) {}

  static Value array() { Value v; v.type_ = Type::Array; return v; }
  static Value object() { Value v; v.type_ = Type::Object; return v; }

  Type type() const { return type_; }
  bool is_null() const { return type_ == Type::Null; }
  bool is_bool() const { return type_ == Type::Bool; }
  bool is_number() const { return type_ == Type::Number; }
  bool is_string() const { return type_ == Type::String; }
  bool is_array() const { return type_ == Type::Array; }
  bool is_object() const { return type_ == Type::Object; }

  bool as_bool() const { expect(Type::Bool, "bool"); return b_; }
  double as_number() const { expect(Type::Number, "number"); return n_; }
  int as_int() const { return static_cast<int>(std::llround(as_number())); }
  const std::string &as_string() const { expect(Type::String, "string"); return s_; }

  // --- arrays ---------------------------------------------------------------------
  const std::vector<Value> &items() const { expect(Type::Array, "array"); return arr_; }
  Value &push_back(Value v) { expect(Type::Array, "array"); arr_.push_back(std::move(v)); return *this; }
  size_t size() const {
    if (type_ == Type::Array) return arr_.size();
    if (type_ == Type::Object) return keys_.size();
    return 0;
  }

  // --- objects (insertion-ordered) -------------------------------------------------
  const std::vector<std::string> &keys() const { expect(Type::Object, "object"); return keys_; }
  const std::vector<Value> &values() const { expect(Type::Object, "object"); return vals_; }
  Value &set(std::string key, Value v) {
    expect(Type::Object, "object");
    for (size_t i = 0; i < keys_.size(); ++i)
      if (keys_[i] == key) { vals_[i] = std::move(v); return *this; }
    keys_.push_back(std::move(key));
    vals_.push_back(std::move(v));
    return *this;
  }
  const Value *find(std::string_view key) const {
    if (type_ != Type::Object) return nullptr;
    for (size_t i = 0; i < keys_.size(); ++i)
      if (keys_[i] == key) return &vals_[i];
    return nullptr;
  }
  bool contains(std::string_view key) const { return find(key) != nullptr; }
  const Value &at(std::string_view key) const {
    const Value *v = find(key);
    if (!v) throw std::runtime_error("json: missing key '" + std::string(key) + "'");
    return *v;
  }

  // --- serialization -------------------------------------------------------------
  std::string dump(int indent = -1) const {
    std::string out;
    dump_to(out, indent, 0);
    return out;
  }

  static Value parse(std::string_view text) {
    Parser p{text, 0};
    p.skip_ws();
    Value v = p.value();
    p.skip_ws();
    if (p.pos != text.size()) p.fail("trailing characters");
    return v;
  }

private:
  Type type_ = Type::Null;
  bool b_ = false;
  double n_ = 0;
  std::string s_;
  std::vector<Value> arr_;
  std::vector<std::string> keys_;
  std::vector<Value> vals_;

  void expect(Type t, const char *name) const {
    if (type_ != t) throw std::runtime_error(std::string("json: value is not a ") + name);
  }

  static void escape(std::string &out, const std::string &s) {
    out.push_back('"');
    for (unsigned char c : s) {
      switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof buf, "\\u%04x", c);
          out += buf;
        } else {
          out.push_back(static_cast<char>(c));
        }
      }
    }
    out.push_back('"');
  }

  static void newline(std::string &out, int indent, int depth) {
    if (indent < 0) return;
    out.push_back('\n');
    out.append(static_cast<size_t>(indent * depth), ' ');
  }

  void dump_to(std::string &out, int indent, int depth) const {
    switch (type_) {
    case Type::Null: out += "null"; break;
    case Type::Bool: out += b_ ? "true" : "false"; break;
    case Type::Number: {
      if (!std::isfinite(n_)) { out += "null"; break; }
      char buf[40];
      if (n_ == std::floor(n_) && std::fabs(n_) < 1e15)
        std::snprintf(buf, sizeof buf, "%.0f", n_);
      else
        std::snprintf(buf, sizeof buf, "%.10g", n_);
      out += buf;
      break;
    }
    case Type::String: escape(out, s_); break;
    case Type::Array:
      if (arr_.empty()) { out += "[]"; break; }
      out.push_back('[');
      for (size_t i = 0; i < arr_.size(); ++i) {
        if (i) out.push_back(',');
        newline(out, indent, depth + 1);
        arr_[i].dump_to(out, indent, depth + 1);
      }
      newline(out, indent, depth);
      out.push_back(']');
      break;
    case Type::Object:
      if (keys_.empty()) { out += "{}"; break; }
      out.push_back('{');
      for (size_t i = 0; i < keys_.size(); ++i) {
        if (i) out.push_back(',');
        newline(out, indent, depth + 1);
        escape(out, keys_[i]);
        out += indent < 0 ? ":" : ": ";
        vals_[i].dump_to(out, indent, depth + 1);
      }
      newline(out, indent, depth);
      out.push_back('}');
      break;
    }
  }

  struct Parser {
    std::string_view s;
    size_t pos;

    [[noreturn]] void fail(const char *msg) const {
      throw std::runtime_error("json parse error at offset " + std::to_string(pos) + ": " + msg);
    }
    void skip_ws() {
      while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\n' || s[pos] == '\t' || s[pos] == '\r')) ++pos;
    }
    char peek() const { return pos < s.size() ? s[pos] : '\0'; }
    void expect_char(char c) {
      if (peek() != c) fail("unexpected character");
      ++pos;
    }
    bool match(std::string_view lit) {
      if (s.substr(pos, lit.size()) == lit) { pos += lit.size(); return true; }
      return false;
    }

    Value value() {
      skip_ws();
      char c = peek();
      if (c == '{') return object();
      if (c == '[') return array();
      if (c == '"') return Value(string());
      if (match("true")) return Value(true);
      if (match("false")) return Value(false);
      if (match("null")) return Value();
      if (c == '-' || (c >= '0' && c <= '9')) return number();
      fail("unexpected token");
    }

    Value number() {
      size_t start = pos;
      if (peek() == '-') ++pos;
      while (pos < s.size() && (std::isdigit(static_cast<unsigned char>(s[pos])) || s[pos] == '.' ||
                                s[pos] == 'e' || s[pos] == 'E' || s[pos] == '+' || s[pos] == '-'))
        ++pos;
      std::string tmp(s.substr(start, pos - start));
      char *end = nullptr;
      double d = std::strtod(tmp.c_str(), &end);
      if (end == tmp.c_str() || *end != '\0') fail("bad number");
      return Value(d);
    }

    static void append_utf8(std::string &out, unsigned cp) {
      if (cp < 0x80) out.push_back(static_cast<char>(cp));
      else if (cp < 0x800) { out.push_back(static_cast<char>(0xC0 | (cp >> 6))); out.push_back(static_cast<char>(0x80 | (cp & 0x3F))); }
      else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
      } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
      }
    }

    unsigned hex4() {
      if (pos + 4 > s.size()) fail("bad \\u escape");
      unsigned v = 0;
      for (int i = 0; i < 4; ++i) {
        char c = s[pos++];
        v <<= 4;
        if (c >= '0' && c <= '9') v |= static_cast<unsigned>(c - '0');
        else if (c >= 'a' && c <= 'f') v |= static_cast<unsigned>(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= static_cast<unsigned>(c - 'A' + 10);
        else fail("bad hex digit");
      }
      return v;
    }

    std::string string() {
      expect_char('"');
      std::string out;
      while (true) {
        if (pos >= s.size()) fail("unterminated string");
        char c = s[pos++];
        if (c == '"') break;
        if (c != '\\') { out.push_back(c); continue; }
        if (pos >= s.size()) fail("bad escape");
        char e = s[pos++];
        switch (e) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'u': {
          unsigned cp = hex4();
          if (cp >= 0xD800 && cp < 0xDC00 && match("\\u")) {
            unsigned lo = hex4();
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
          }
          append_utf8(out, cp);
          break;
        }
        default: fail("unknown escape");
        }
      }
      return out;
    }

    Value array() {
      expect_char('[');
      Value v = Value::array();
      skip_ws();
      if (peek() == ']') { ++pos; return v; }
      while (true) {
        v.push_back(value());
        skip_ws();
        if (peek() == ',') { ++pos; continue; }
        expect_char(']');
        return v;
      }
    }

    Value object() {
      expect_char('{');
      Value v = Value::object();
      skip_ws();
      if (peek() == '}') { ++pos; return v; }
      while (true) {
        skip_ws();
        std::string k = string();
        skip_ws();
        expect_char(':');
        v.set(std::move(k), value());
        skip_ws();
        if (peek() == ',') { ++pos; continue; }
        expect_char('}');
        return v;
      }
    }
  };
};

using Array = std::vector<Value>;

} // namespace ptxforge::json
