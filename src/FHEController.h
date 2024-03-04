//
// Created by Lorenzo on 12/01/24.
//

#ifndef NEWBERT_FHECONTROLLER_H
#define NEWBERT_FHECONTROLLER_H

#include "openfhe.h"
#include "ciphertext-ser.h"
#include "scheme/ckksrns/ckksrns-ser.h"
#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include <thread>
#include "Utils.h"

using namespace lbcrypto;
using namespace std;
using namespace std::chrono;

using namespace utils;

using Ptxt = Plaintext;
using Ctxt = Ciphertext<DCRTPoly>;

class FHEController {
    CryptoContext<DCRTPoly> context;

public:
    int circuit_depth;
    int num_slots;

    FHEController() {}

    /*
     * Context generating/loading stuff
     */
    void generate_context(bool serialize = false, bool secure = false);
    void generate_context(int log_ring, int log_scale, int log_primes, int digits_hks, int cts_levels, int stc_levels, int relu_deg, bool serialize = false);
    void load_context(bool verbose = true);
    void test_context();

    /*
     * Generating bootstrapping and rotation keys stuff
     */
    void generate_bootstrapping_keys(int bootstrap_slots);
    void generate_rotation_keys(vector<int> rotations, bool serialize = false, string filename = "");
    void generate_bootstrapping_and_rotation_keys(vector<int> rotations,
                                                  int bootstrap_slots,
                                                  bool serialize,
                                                  const string& filename);


    void load_bootstrapping_and_rotation_keys(const string& filename, int bootstrap_slots, bool verbose);
    void load_rotation_keys(const string& filename, bool verbose);
    void clear_bootstrapping_and_rotation_keys(int bootstrap_num_slots);
    void clear_rotation_keys();
    void clear_context(int bootstrapping_key_slots);


    /*
     * CKKS Encoding/Decoding/Encryption/Decryption
     */
    Ptxt encode(const vector<double>& vec, int level, int plaintext_num_slots);
    Ptxt encode(double val, int level, int plaintext_num_slots);
    Ctxt encrypt(const vector<double>& vec, int level = 0, int plaintext_num_slots = 0);
    Ctxt encrypt_ptxt(const Ptxt& p);
    Ptxt decrypt(const Ctxt& c);
    vector<double> decrypt_tovector(const Ctxt& c, int slots);

    /*
     * Homomorphic operations
     */
    Ctxt add(const Ctxt& c1, const Ctxt& c2);
    Ctxt add(const Ctxt& c1, const Ptxt& c2);
    Ctxt add(vector<Ctxt> c);
    Ctxt mult(const Ctxt& c1, const Ctxt& c2);
    Ctxt mult(const Ctxt& c, double d);
    Ctxt mult(const Ctxt& c, const Ptxt& p);
    Ctxt rotate(const Ctxt& c, int index);
    Ctxt bootstrap(const Ctxt& c, bool timing = false);
    Ctxt bootstrap(const Ctxt& c, int precision, bool timing = false);
    Ctxt relu(const Ctxt& c, double scale, bool timing = false);
    Ctxt relu_wide(const Ctxt& c, double a, double b, int degree, double scale, bool timing = false);

    /*
     * I/O
     */
    Ctxt read_input(const string& filename, double scale = 1);
    Ctxt read_repeated_input(const string& filename, double scale = 1);
    Ctxt read_expanded_input(const string& filename, double scale = 1);

    Ptxt read_plain_input(const string& filename, int level = 0, double scale = 1);
    //Ptxt read_plain_512_input(const string& filename, int level = 0, double scale = 1);
    Ptxt read_plain_repeated_input(const string& filename, int level = 0, double scale = 1);
    Ptxt read_plain_repeated_512_input(const string& filename, int level = 0, double scale = 1);
    Ptxt read_plain_expanded_input(const string& filename, int level = 0, double scale = 1);
    Ptxt read_plain_expanded_input(const string& filename, int level, double scale, int num_inputs);


    void print(const Ctxt& c, int slots = 0, string prefix = "");
    void print_padded(const Ctxt& c, int slots = 0, int padding = 1, string prefix = "");
    void print_expanded(const Ctxt& c, int slots = 0, int expansion_factor = 1, string prefix = "");
    void print_min_max(const Ctxt& c);

    Ctxt rotsum(const Ctxt &in, int slots, int padding);
    Ctxt rotsum_padded(const Ctxt &in, int slots);

    Ctxt repeat(const Ctxt &in, int slots);
    Ctxt repeat(const Ctxt &in, int slots, int padding);

    vector<Ctxt> matmulRE(vector<Ctxt> rows, const Ptxt& weight, const Ptxt& bias );
    vector<Ctxt> matmulRE(vector<Ctxt> rows, const Ptxt& weight, const Ptxt& bias, int row_size, int padding );
    vector<Ctxt> matmulRE(vector<Ctxt> rows, const Ctxt& weight, int row_size, int padding );
    vector<Ctxt> matmulRElarge(vector<Ctxt>& rows, const vector<Ptxt>& weight, const Ptxt& bias, double mask_value = 1);
    vector<Ctxt> matmulCR(vector<Ctxt> rows, const Ptxt& weight, const Ptxt& bias );
    vector<Ctxt> matmulCR(vector<Ctxt> rows, const Ctxt& matrix);
    vector<Ctxt> matmulCRlarge(vector<vector<Ctxt>> rows, vector<Ptxt> weights, const Ptxt& bias);

    Ctxt matmulScores(vector<Ctxt> queries, const Ctxt& key);

    Ctxt wrapUpRepeated(vector<Ctxt> vectors);
    Ctxt wrapUpExpanded(vector<Ctxt> vectors);
    vector<Ctxt> unwrapExpanded(Ctxt c, int inputs_num);
    vector<vector<Ctxt>> unwrapRepeatedLarge(vector<Ctxt> c, int input_number);
    vector<Ctxt> unwrapScoresExpanded(Ctxt c, int inputs_num);
    vector<Ctxt> unwrap_512_in_4_128(const Ctxt& c, int index);

    vector<Ctxt> generate_containers(vector<Ctxt> inputs, const Ptxt& bias);
    Ctxt wrap_containers(vector<Ctxt> inputs, int inputs_number);

    Ctxt mask_block(const Ctxt& c, int from, int to, double mask_value = 1);
    Ctxt mask_heads(const Ctxt& c, double mask_value = 1);
    Ctxt mask_mod_n(const Ctxt& c, int n);
    Ctxt mask_mod_n(const Ctxt& c, int n, int padding, int max_slots);
    Ctxt mask_first_n(const Ctxt& c, int n, double mask_value = 1);

    Ctxt eval_exp(const Ctxt& c, int inputs_number);
    Ctxt eval_inverse(const Ctxt& c, double min, double max);
    Ctxt eval_inverse_naive(const Ctxt& c, double min, double max);
    Ctxt eval_inverse_naive_2(const Ctxt& c, double min, double max, double mult);
    Ctxt eval_gelu_function(const Ctxt& c, double min, double max, double mult, int degree);
    Ctxt eval_tanh_function(const Ctxt& c, double min, double max, double mult, int degree);

    vector<Ctxt> slicing(vector<Ctxt> &arr, int X, int Y);

    void save(Ctxt v, string filename);
    void save(vector<Ctxt> v, string filename);
    vector<Ctxt> load_vector(string filename);
    Ctxt load_ciphertext(string filename);

    int relu_degree = 119;
    string parameters_folder = "keys";

private:
    KeyPair<DCRTPoly> key_pair;
    vector<uint32_t> level_budget = {4, 4};


};


#endif //NEWBERT_FHECONTROLLER_H
