//
// Created by Lorenzo on 24/10/23.
//

#include "FHEController.h"

void FHEController::generate_context(bool serialize) {
    CCParams<CryptoContextCKKSRNS> parameters;

    num_slots = 1 << 14;

    parameters.SetSecretKeyDist(SPARSE_TERNARY);
   // parameters.SetSecurityLevel(lbcrypto::HEStd_128_classic);
    parameters.SetSecurityLevel(lbcrypto::HEStd_NotSet);
    parameters.SetNumLargeDigits(3); //d_{num} Se lo riduci, aumenti il logQP, se lo aumenti, aumenti memori
    //parameters.SetRingDim(1 << 16);
    parameters.SetRingDim(1 << 15);
    parameters.SetBatchSize(num_slots);

    level_budget = {3, 3};

    ScalingTechnique rescaleTech = FLEXIBLEAUTO;

    int dcrtBits               = 59;
    int firstMod               = 60;

    parameters.SetScalingModSize(dcrtBits);
    parameters.SetScalingTechnique(rescaleTech);
    parameters.SetFirstModSize(firstMod);

    uint32_t approxBootstrapDepth = 4 + 4;

    uint32_t levelsUsedBeforeBootstrap = 12;

    circuit_depth = levelsUsedBeforeBootstrap + FHECKKSRNS::GetBootstrapDepth(approxBootstrapDepth, level_budget, SPARSE_TERNARY);

    cout << endl << "Ciphertexts depth: " << circuit_depth << ", available multiplications: " << levelsUsedBeforeBootstrap - 2 << endl;

    parameters.SetMultiplicativeDepth(circuit_depth);

    context = GenCryptoContext(parameters);

    cout << "Context built, generating keys..." << endl;

    context->Enable(PKE);
    context->Enable(KEYSWITCH);
    context->Enable(LEVELEDSHE);
    context->Enable(ADVANCEDSHE);
    context->Enable(FHE);

    key_pair = context->KeyGen();

    context->EvalMultKeyGen(key_pair.secretKey);

    cout << "Generated." << endl;

    if (!serialize) {
        return;
    }

    cout << "Now serializing keys ..." << endl;

    ofstream multKeyFile("../" + parameters_folder + "/mult-keys.txt", ios::out | ios::binary);
    if (multKeyFile.is_open()) {
        if (!context->SerializeEvalMultKey(multKeyFile, SerType::BINARY)) {
            cerr << "Error writing eval mult keys" << std::endl;
            exit(1);
        }
        cout << "Relinearization Keys have been serialized" << std::endl;
        multKeyFile.close();
    }
    else {
        cerr << "Error serializing EvalMult keys in \"" << "../" + parameters_folder + "/mult-keys.txt" << "\"" << endl;
        exit(1);
    }

    if (!Serial::SerializeToFile("../" + parameters_folder + "/crypto-context.txt", context, SerType::BINARY)) {
        cerr << "Error writing serialization of the crypto context to crypto-context.txt" << endl;
    } else {
        cout << "Crypto Context have been serialized" << std::endl;
    }

    if (!Serial::SerializeToFile("../" + parameters_folder + "/public-key.txt", key_pair.publicKey, SerType::BINARY)) {
        cerr << "Error writing serialization of public key to public-key.txt" << endl;
    } else {
        cout << "Public Key has been serialized" << std::endl;
    }

    if (!Serial::SerializeToFile("../" + parameters_folder + "/secret-key.txt", key_pair.secretKey, SerType::BINARY)) {
        cerr << "Error writing serialization of public key to secret-key.txt" << endl;
    } else {
        cout << "Secret Key has been serialized" << std::endl;
    }
}

void FHEController::generate_context(int log_ring, int log_scale, int log_primes, int digits_hks, int cts_levels,
                                     int stc_levels, int relu_deg, bool serialize) {

    CCParams<CryptoContextCKKSRNS> parameters;

    num_slots = 1 << 14;

    parameters.SetSecretKeyDist(SPARSE_TERNARY);
    //parameters.SetSecurityLevel(lbcrypto::HEStd_128_classic);
    parameters.SetSecurityLevel(lbcrypto::HEStd_NotSet);
    parameters.SetNumLargeDigits(digits_hks);
    parameters.SetRingDim(1 << log_ring);
    parameters.SetBatchSize(num_slots);

    level_budget = vector<uint32_t>();

    level_budget.push_back(cts_levels);
    level_budget.push_back(stc_levels);

    int dcrtBits = log_primes;
    int firstMod = log_scale;

    parameters.SetScalingModSize(dcrtBits);
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetFirstModSize(firstMod);

    uint32_t approxBootstrapDepth = 4 + 4; //During EvalRaise, Chebyshev, DoubleAngle

    uint32_t levelsUsedBeforeBootstrap = 12;

    circuit_depth = levelsUsedBeforeBootstrap +
                    FHECKKSRNS::GetBootstrapDepth(approxBootstrapDepth, level_budget, SPARSE_TERNARY);

    cout << endl << "Ciphertexts depth: " << circuit_depth << ", available multiplications: "
         << levelsUsedBeforeBootstrap - 2 << endl;

    parameters.SetMultiplicativeDepth(circuit_depth);

    context = GenCryptoContext(parameters);

    cout << "Context built, generating keys..." << endl;

    context->Enable(PKE);
    context->Enable(KEYSWITCH);
    context->Enable(LEVELEDSHE);
    context->Enable(ADVANCEDSHE);
    context->Enable(FHE);

    key_pair = context->KeyGen();

    context->EvalMultKeyGen(key_pair.secretKey);

    cout << "Generated." << endl;

    if (!serialize) {
        return;
    }

    cout << "Now serializing keys ..." << endl;

    ofstream multKeyFile("../" + parameters_folder + "/mult-keys.txt", ios::out | ios::binary);
    if (multKeyFile.is_open()) {
        if (!context->SerializeEvalMultKey(multKeyFile, SerType::BINARY)) {
            cerr << "Error writing EvalMult keys" << std::endl;
            exit(1);
        }
        cout << "EvalMult keys have been serialized" << std::endl;
        multKeyFile.close();
    } else {
        cerr << "Error serializing EvalMult keys in \"" << "../" + parameters_folder + "/mult-keys.txt" << "\"" << endl;
        exit(1);
    }

    if (!Serial::SerializeToFile("../" + parameters_folder + "/crypto-context.txt", context, SerType::BINARY)) {
        cerr << "Error writing serialization of the crypto context to crypto-context.txt" << endl;
    } else {
        cout << "Crypto Context have been serialized" << std::endl;
    }

    if (!Serial::SerializeToFile("../" + parameters_folder + "/public-key.txt", key_pair.publicKey, SerType::BINARY)) {
        cerr << "Error writing serialization of public key to public-key.txt" << endl;
    } else {
        cout << "Public Key has been serialized" << std::endl;
    }

    if (!Serial::SerializeToFile("../" + parameters_folder + "/secret-key.txt", key_pair.secretKey, SerType::BINARY)) {
        cerr << "Error writing serialization of public key to secret-key.txt" << endl;
    } else {
        cout << "Secret Key has been serialized" << std::endl;
    }
}

void FHEController::load_context(bool verbose) {
    context->ClearEvalMultKeys();
    context->ClearEvalAutomorphismKeys();

    CryptoContextFactory<lbcrypto::DCRTPoly>::ReleaseAllContexts();

    if (verbose) cout << "Reading serialized context..." << endl;

    if (!Serial::DeserializeFromFile("../" + parameters_folder + "/crypto-context.txt", context, SerType::BINARY)) {
        cerr << "I cannot read serialized data from: " << "../" + parameters_folder + "/crypto-context.txt" << endl;
        exit(1);
    }

    PublicKey<DCRTPoly> clientPublicKey;
    if (!Serial::DeserializeFromFile("../" + parameters_folder + "/public-key.txt", clientPublicKey, SerType::BINARY)) {
        cerr << "I cannot read serialized data from public-key.txt" << endl;
        exit(1);
    }

    PrivateKey<DCRTPoly> serverSecretKey;
    if (!Serial::DeserializeFromFile("../" + parameters_folder + "/secret-key.txt", serverSecretKey, SerType::BINARY)) {
        cerr << "I cannot read serialized data from public-key.txt" << endl;
        exit(1);
    }

    key_pair.publicKey = clientPublicKey;
    key_pair.secretKey = serverSecretKey;

    std::ifstream multKeyIStream("../" + parameters_folder + "/mult-keys.txt", ios::in | ios::binary);
    if (!multKeyIStream.is_open()) {
        cerr << "Cannot read serialization from " << "mult-keys.txt" << endl;
        exit(1);
    }
    if (!context->DeserializeEvalMultKey(multKeyIStream, SerType::BINARY)) {
        cerr << "Could not deserialize eval mult key file" << endl;
        exit(1);
    }

    level_budget = {3, 3};

    if (verbose) cout << "CtoS: " << level_budget[0] << ", StoC: " << level_budget[1] << endl;

    uint32_t approxBootstrapDepth = 8;

    uint32_t levelsUsedBeforeBootstrap = 12;

    circuit_depth = levelsUsedBeforeBootstrap + FHECKKSRNS::GetBootstrapDepth(approxBootstrapDepth, level_budget, SPARSE_TERNARY);

    //if (verbose) cout << "Circuit depth: " << circuit_depth << ", available multiplications: " << levelsUsedBeforeBootstrap - 2 << endl;
    cout << "Circuit depth: " << circuit_depth << ", available multiplications: " << levelsUsedBeforeBootstrap - 2 << endl;

    num_slots = 1 << 14;
}


void FHEController::generate_bootstrapping_keys(int bootstrap_slots) {
    context->EvalBootstrapSetup(level_budget, {0, 0}, bootstrap_slots);
    context->EvalBootstrapKeyGen(key_pair.secretKey, bootstrap_slots);
}

void FHEController::generate_rotation_keys(vector<int> rotations, bool serialize, std::string filename) {
    if (serialize && filename.size() == 0) {
        cout << "Filename cannot be empty when serializing rotation keys." << endl;
        return;
    }

    context->EvalRotateKeyGen(key_pair.secretKey, rotations);

    if (serialize) {
        ofstream rotationKeyFile("../" + parameters_folder + "/rot_" + filename, ios::out | ios::binary);
        if (rotationKeyFile.is_open()) {
            if (!context->SerializeEvalAutomorphismKey(rotationKeyFile, SerType::BINARY)) {
                cerr << "Error writing rotation keys" << std::endl;
                exit(1);
            }
            cout << "Rotation keys \"" << filename << "\" have been serialized" << std::endl;
        } else {
            cerr << "Error serializing Rotation keys" << "../" + parameters_folder + "/rot_" + filename << std::endl;
            exit(1);
        }
    }
}

void FHEController::generate_bootstrapping_and_rotation_keys(vector<int> rotations, int bootstrap_slots, bool serialize, const string& filename) {
    if (serialize && filename.empty()) {
        cout << "Filename cannot be empty when serializing bootstrapping and rotation keys." << endl;
        return;
    }

    generate_bootstrapping_keys(bootstrap_slots);
    generate_rotation_keys(rotations, serialize, filename);
}

void FHEController::load_bootstrapping_and_rotation_keys(const string& filename, int bootstrap_slots, bool verbose) {
    if (verbose) cout << endl << "Loading bootstrapping and rotations keys from " << filename << "..." << endl;

    auto start = start_time();

    context->EvalBootstrapSetup(level_budget, {0, 0}, bootstrap_slots);

    if (verbose)  cout << "(1/2) Bootstrapping precomputations completed!" << endl;


    ifstream rotKeyIStream("../" + parameters_folder + "/rot_" + filename, ios::in | ios::binary);
    if (!rotKeyIStream.is_open()) {
        cerr << "Cannot read serialization from " << "../" + parameters_folder + "/" << "rot_" << filename << std::endl;
        exit(1);
    }

    if (!context->DeserializeEvalAutomorphismKey(rotKeyIStream, SerType::BINARY)) {
        cerr << "Could not deserialize eval rot key file" << std::endl;
        exit(1);
    }

    if (verbose) cout << "(2/2) Rotation keys read!" << endl;

    if (verbose) print_duration(start, "Loading bootstrapping pre-computations + rotations");

    if (verbose) cout << endl;
}

void FHEController::load_rotation_keys(const string& filename, bool verbose) {
    if (verbose) cout << endl << "Loading rotations keys from " << filename << "..." << endl;

    auto start = start_time();

    ifstream rotKeyIStream("../" + parameters_folder + "/rot_" + filename, ios::in | ios::binary);
    if (!rotKeyIStream.is_open()) {
        cerr << "Cannot read serialization from " << "../" + parameters_folder + "/" << "rot_" << filename << std::endl;
        exit(1);
    }

    if (!context->DeserializeEvalAutomorphismKey(rotKeyIStream, SerType::BINARY)) {
        cerr << "Could not deserialize eval rot key file" << std::endl;
        exit(1);
    }

    if (verbose) {
        cout << "(1/1) Rotation keys read!" << endl;
        print_duration(start, "Loading rotation keys");
        cout << endl;
    }
}

void FHEController::clear_bootstrapping_and_rotation_keys(int bootstrap_num_slots) {
    FHECKKSRNS* derivedPtr = dynamic_cast<FHECKKSRNS*>(context->GetScheme()->GetFHE().get());
    derivedPtr->m_bootPrecomMap.erase(bootstrap_num_slots);
    clear_rotation_keys();
}

void FHEController::clear_rotation_keys() {
    context->ClearEvalAutomorphismKeys();
}

void FHEController::clear_context(int bootstrapping_key_slots) {
    if (bootstrapping_key_slots != 0)
        clear_bootstrapping_and_rotation_keys(bootstrapping_key_slots);
    else
        clear_rotation_keys();

    context->ClearEvalMultKeys();
}

/*
 * CKKS Encoding/Decoding/Encryption/Decryption
 */
Ptxt FHEController::encode(const vector<double> &vec, int level, int plaintext_num_slots) {
    if (plaintext_num_slots == 0) {
        plaintext_num_slots = num_slots;
    }

    Ptxt p = context->MakeCKKSPackedPlaintext(vec, 1, level, nullptr, plaintext_num_slots);
    p->SetLength(plaintext_num_slots);
    return p;
}

Ptxt FHEController::encode(double val, int level, int plaintext_num_slots) {
    if (plaintext_num_slots == 0) {
        plaintext_num_slots = num_slots;
    }

    vector<double> vec;
    for (int i = 0; i < plaintext_num_slots; i++) {
        vec.push_back(val);
    }

    Ptxt p = context->MakeCKKSPackedPlaintext(vec, 1, level, nullptr, plaintext_num_slots);
    p->SetLength(plaintext_num_slots);
    return p;
}

Ctxt FHEController::encrypt(const vector<double> &vec, int level, int plaintext_num_slots) {
    if (plaintext_num_slots == 0) {
        plaintext_num_slots = num_slots;
    }

    Ptxt p = encode(vec, level, plaintext_num_slots);

    return context->Encrypt(p, key_pair.publicKey);
}

Ctxt FHEController::encrypt_ptxt(const Ptxt& p) {
    return context->Encrypt(p, key_pair.publicKey);
}

Ptxt FHEController::decrypt(const Ctxt &c) {
    Ptxt p;
    context->Decrypt(key_pair.secretKey, c, &p);
    return p;
}

vector<double> FHEController::decrypt_tovector(const Ctxt &c, int slots) {
    if (slots == 0) {
        slots = num_slots;
    }

    Ptxt p;
    context->Decrypt(key_pair.secretKey, c, &p);
    p->SetSlots(slots);
    p->SetLength(slots);
    vector<double> vec = p->GetRealPackedValue();
    return vec;
}

/*
 * Homomorphic operations
 */
Ctxt FHEController::add(const Ctxt &c1, const Ctxt &c2) {
    return context->EvalAdd(c1, c2);
}

Ctxt FHEController::add(const Ctxt &c1, const Ptxt &c2) {
    return context->EvalAdd(c1, c2);
}

Ctxt FHEController::add(vector<Ctxt> c) {
    return context->EvalAddMany(c);
}

Ctxt FHEController::mult(const Ctxt &c1, double d) {
    Ptxt p = encode(d, c1->GetLevel(), num_slots);
    return context->EvalMult(c1, p);
}

Ctxt FHEController::mult(const Ctxt &c, const Ptxt& p) {
    return context->EvalMult(c, p);
}

Ctxt FHEController::mult(const Ctxt &c1, const Ctxt& c2) {
    return context->EvalMult(c1, c2);
}

Ctxt FHEController::rotate(const Ctxt &c, int index) {
    return context->EvalRotate(c, index);
}

Ctxt FHEController::bootstrap(const Ctxt &c, bool timing) {
    if (static_cast<int>(c->GetLevel()) + 2 < circuit_depth && timing) {
        cout << "You are bootstrapping with remaining levels! You are at " << to_string(c->GetLevel()) << "/" << circuit_depth - 2 << endl;
    }


    auto start = start_time();

    Ctxt res = context->EvalBootstrap(c);

    if (timing) {
        print_duration(start, "Bootstrapping " + to_string(c->GetSlots()) + " slots");
    }

    return res;
}

Ctxt FHEController::bootstrap(const Ctxt &c, int precision, bool timing) {
    if (static_cast<int>(c->GetLevel()) + 2 < circuit_depth) {
        cout << "You are bootstrapping with remaining levels! You are at " << to_string(c->GetLevel()) << "/" << circuit_depth - 2 << endl;
    }

    auto start = start_time();

    Ctxt res = context->EvalBootstrap(c, 2, precision);

    if (timing) {
        print_duration(start, "Double Bootstrapping " + to_string(c->GetSlots()) + " slots");
    }


    return res;
}

Ctxt FHEController::relu(const Ctxt &c, double scale, bool timing) {
    auto start = start_time();

    /*
     * Max min
     */
    Ptxt result;
    context->Decrypt(key_pair.secretKey, c, &result);
    vector<double> v = result->GetRealPackedValue();

    //cout << "min: " << *min_element(v.begin(), v.end()) << ", max: " << *max_element(v.begin(), v.end()) << endl;
    /*
     * Max min
     */

    Ctxt res = context->EvalChebyshevFunction([scale](double x) -> double { if (x < 0) return 0; else return (1 / scale) * x; }, c,
                                              -1,
                                              1, relu_degree);

    if (timing) {
        print_duration(start, "ReLU d = " + to_string(relu_degree) + " evaluation");
    }

    return res;
}

Ctxt FHEController::relu_wide(const Ctxt &c, double a, double b, int degree, double scale, bool timing) {
    auto start = start_time();

    /*
     * Max min
     */
    Ptxt result;
    context->Decrypt(key_pair.secretKey, c, &result);
    vector<double> v = result->GetRealPackedValue();

    cout << "min: " << *min_element(v.begin(), v.end()) << ", max: " << *max_element(v.begin(), v.end()) << endl;
    /*
     * Max min
     */

    Ctxt res = context->EvalChebyshevFunction([scale](double x) -> double { if (x < 0) return 0; else return (1 / scale) * x; }, c,
                                              a,
                                              b, degree);
    if (timing) {
        print_duration(start, "ReLU d = " + to_string(degree) + " evaluation");
    }

    return res;
}


/*
 * I/O
 */

Ctxt FHEController::read_input(const string& filename, double scale) {
    vector<double> input = read_values_from_file(filename);

    int size = static_cast<int>(input.size());

    if (scale != 1) {
        for (int i = 0; i < size; i++) {
            input[i] = input[i] * scale;
        }
    }

    return context->Encrypt(key_pair.publicKey, context->MakeCKKSPackedPlaintext(input, 1, circuit_depth - 10, nullptr, num_slots));
}

Ptxt FHEController::read_plain_input(const string& filename, int level, double scale) {
    vector<double> input = read_values_from_file(filename);

    int size = static_cast<int>(input.size());

    if (scale != 1) {
        for (int i = 0; i < size; i++) {
            input[i] = input[i] * scale;
        }
    }

    return context->MakeCKKSPackedPlaintext(input, 1, level, nullptr, num_slots);
}

Ctxt FHEController::read_repeated_input(const string& filename, double scale) {
    //Assumption: inputs have 128 values
    vector<double> input = read_values_from_file(filename);

    vector<double> repeated;

    for (int j = 0; j < 128; j++) {
        for (int i = 0; i < 128; i++) {
            repeated.push_back(input[i]);
        }
    }

    int size = static_cast<int>(input.size());

    if (scale != 1) {
        for (int i = 0; i < size; i++) {
            input[i] = input[i] * scale;
        }
    }

    return context->Encrypt(key_pair.publicKey, context->MakeCKKSPackedPlaintext(input, 1, 0, nullptr, num_slots));
}

Ptxt FHEController::read_plain_repeated_input(const string& filename, int level, double scale) {
    //Assumption: inputs have 128 values
    vector<double> input = read_values_from_file(filename);

    vector<double> repeated;

    for (int j = 0; j < 128; j++) {
        for (int i = 0; i < 128; i++) {
            repeated.push_back(input[i]);
        }
    }

    int size = static_cast<int>(repeated.size());

    if (scale != 1) {
        for (int i = 0; i < size; i++) {
            repeated[i] = repeated[i] * scale;
        }
    }

    return context->MakeCKKSPackedPlaintext(repeated, 1, level, nullptr, num_slots);
}

Ptxt FHEController::read_plain_repeated_512_input(const string& filename, int level, double scale) {
    //Assumption: inputs have 128 values
    vector<double> input = read_values_from_file(filename);

    vector<double> repeated;

    for (int j = 0; j < 32; j++) {
        for (int i = 0; i < 512; i++) {
            repeated.push_back(input[i]);
        }
    }

    int size = static_cast<int>(repeated.size());

    if (scale != 1) {
        for (int i = 0; i < size; i++) {
            repeated[i] = repeated[i] * scale;
        }
    }

    return context->MakeCKKSPackedPlaintext(repeated, 1, level, nullptr, num_slots);
}

Ctxt FHEController::read_expanded_input(const string& filename, double scale) {
    //Assumption: inputs have 128 values
    vector<double> input = read_values_from_file(filename);

    vector<double> repeated;

    for (int j = 0; j < 128; j++) {
        for (int i = 0; i < 128; i++) {
            repeated.push_back(input[j]);
        }
    }

    int size = static_cast<int>(repeated.size());

    if (scale != 1) {
        for (int i = 0; i < size; i++) {
            repeated[i] = repeated[i] * scale;
        }
    }

    return context->Encrypt(key_pair.publicKey, context->MakeCKKSPackedPlaintext(repeated, 1, 0, nullptr, num_slots));
}

Ptxt FHEController::read_plain_expanded_input(const string& filename, int level, double scale) {
    //Assumption: inputs have 128 values
    vector<double> input = read_values_from_file(filename);

    vector<double> repeated;

    for (int j = 0; j < 128; j++) {
        for (int i = 0; i < 128; i++) {
            repeated.push_back(input[j]);
        }
    }

    int size = static_cast<int>(repeated.size());

    if (scale != 1) {
        for (int i = 0; i < size; i++) {
            repeated[i] = repeated[i] * scale;
        }
    }

    return context->MakeCKKSPackedPlaintext(repeated, 1, level, nullptr, num_slots);
}

Ptxt FHEController::read_plain_expanded_input(const string& filename, int level, double scale, int num_inputs) {
    //Assumption: inputs have 128 values
    vector<double> input = read_values_from_file(filename);

    vector<double> repeated;

    for (int j = 0; j < 128; j++) {
        for (int i = 0; i < num_inputs; i++) {
            repeated.push_back(input[j]);
        }
        for (int i = 0; i < 128 - num_inputs; i++) {
            repeated.push_back(0);
        }
    }

    int size = static_cast<int>(repeated.size());

    if (scale != 1) {
        for (int i = 0; i < size; i++) {
            repeated[i] = repeated[i] * scale;
        }
    }

    return context->MakeCKKSPackedPlaintext(repeated, 1, level, nullptr, num_slots);
}

void FHEController::print(const Ctxt &c, int slots, string prefix) {
    if (slots == 0) {
        slots = num_slots;
    }

    cout << prefix << " (Lv. " << c->GetLevel() << ") ";

    Ptxt result;
    context->Decrypt(key_pair.secretKey, c, &result);
    result->SetSlots(num_slots);
    vector<double> v = result->GetRealPackedValue();

    cout << setprecision(4) << fixed;
    cout << "[ ";

    for (int i = 0; i < slots; i += 1) {
        string segno = "";
        if (v[i] > 0) {
            segno = " ";
        } else {
            segno = "-";
            v[i] = -v[i];
        }


        if (i == slots - 1) {
            cout << segno << v[i] << " ]";
        } else {
            if (abs(v[i]) < 0.00000001)
                cout << " 0.0000" << ", ";
            else
                cout << segno << v[i] << ", ";
        }
    }

    cout << endl;
}

void FHEController::print_expanded(const Ctxt &c, int slots, int expansion_factor, string prefix) {
    if (slots == 0) {
        slots = num_slots;
    }

    cout << prefix << " (Lv. " << c->GetLevel() << ") ";

    Ptxt result;
    context->Decrypt(key_pair.secretKey, c, &result);
    result->SetSlots(num_slots);
    vector<double> v = result->GetRealPackedValue();


    cout << setprecision(4) << fixed;
    cout << "[ ";

    for (int i = 0; i < slots; i += 1) {
        if (i % expansion_factor != 0) {
            continue;
        }
        string segno = "";
        if (v[i] > 0) {
            segno = " ";
        } else {
            segno = "-";
            v[i] = -v[i];
        }


        if (i == slots - 1) {
            cout << segno << v[i] << " ]";
        } else {
            if (abs(v[i]) < 0.00000001)
                cout << " 0.000" << ", ";
            else
                cout << segno << v[i] << ", ";
        }
    }

    cout << " ]";

    cout << endl;
}

void FHEController::print_padded(const Ctxt &c, int slots, int padding, string prefix) {
    if (slots == 0) {
        slots = num_slots;
    }

    cout << prefix;

    Ptxt result;
    context->Decrypt(key_pair.secretKey, c, &result);
    result->SetSlots(num_slots);
    vector<double> v = result->GetRealPackedValue();

    cout << setprecision(10) << fixed;
    cout << "[ ";

    for (int i = 0; i < slots * padding; i += padding) {
        string segno = "";
        if (v[i] > 0) {
            segno = " ";
        } else {
            segno = "-";
            v[i] = -v[i];
        }


        if (i == slots - 1) {
            cout << segno << v[i] << " ]";
        } else {
            if (abs(v[i]) < 0.00000001)
                cout << " 0.000" << ", ";
            else
                cout << segno << v[i] << ", ";
        }
    }

    cout << endl;
}

void FHEController::print_min_max(const Ctxt &c) {
    Ptxt result;
    context->Decrypt(key_pair.secretKey, c, &result);
    vector<double> v = result->GetRealPackedValue();

    cout << "min: " << *min_element(v.begin(), v.end()) << ", max: " << *max_element(v.begin(), v.end()) << endl;
}


Ctxt FHEController::rotsum(const Ctxt &in, int slots, int padding) {
    Ctxt result = in->Clone();

    for (int i = 0; i < log2(slots); i++) {
        result = add(result, context->EvalRotate(result, padding * pow(2, i)));
    }

    return result;
}

Ctxt FHEController::rotsum_padded(const Ctxt &in, int slots) {
    Ctxt result = in->Clone();

    for (int i = 0; i < log2(slots); i++) {
        result = add(result, context->EvalRotate(result, slots * pow(2, i)));
    }

    return result;
}

Ctxt FHEController::repeat(const Ctxt &in, int slots) {
    Ctxt res = in->Clone();

    for (int i = 0; i < log2(slots); i++) {
        res = add(res, rotate(res, -pow(2, i)));
    }

    return res;
}

Ctxt FHEController::repeat(const Ctxt &in, int slots, int padding) {
    Ctxt res = in->Clone();

    for (int i = 0; i < log2(slots); i++) {
        res = add(res, rotate(res, padding * (-pow(2, i))));
    }

    return res;
}

vector<Ctxt> FHEController::matmulRE(vector<Ctxt> rows, const Ptxt &weight, const Ptxt &bias) {
    vector<Ctxt> columns;

    for (int i = 0; i < rows.size(); i++) {
        Ctxt m = mult(rows[i], weight);

        m = rotsum(m, 128, 128);

        if (bias != nullptr) m = add(m, bias);

        columns.push_back(m);
    }

    return columns;
}

vector<Ctxt> FHEController::matmulRE(vector<Ctxt> rows, const Ptxt &weight, const Ptxt &bias, int row_size, int padding) {
    vector<Ctxt> columns;

    for (int i = 0; i < rows.size(); i++) {
        Ctxt m = mult(rows[i], weight);

        m = rotsum(m, row_size, padding);

        if (bias != nullptr) m = add(m, bias);

        columns.push_back(m);
    }

    return columns;
}

vector<Ctxt> FHEController::matmulRE(vector<Ctxt> rows, const Ctxt &weight, int row_size, int padding) {
    vector<Ctxt> columns;

    for (int i = 0; i < rows.size(); i++) {
        Ctxt m = mult(rows[i], weight);

        m = rotsum(m, row_size, padding);

        columns.push_back(m);
    }

    return columns;
}

vector<Ctxt> FHEController::matmulRElarge(vector<Ctxt>& inputs, const vector<Ptxt> &weights, const Ptxt &bias, double mask_val) {
    vector<Ctxt> densed;

    for (int i = 0; i < inputs.size(); i++) {
        Ctxt i_th_result;
        for (int j = weights.size() - 1; j >= 0; j--) {
            Ctxt out = mult(inputs[i], weights[j]);
            out = rotsum(out, 128, 128);

            out = mask_first_n(out, 128, mask_val);

            if (j == weights.size() - 1)
                i_th_result = out;
            else {
                //i_th_result = rotate(i_th_result, -128);
                i_th_result = rotate(i_th_result, -64);
                i_th_result = rotate(i_th_result, -64);

                i_th_result = add(i_th_result, out);
            }

        }

        densed.push_back(i_th_result);
    }

    return densed;
}

vector<Ctxt> FHEController::matmulCR(vector<Ctxt> rows, const Ctxt& matrix) {
    vector<Ctxt> columns;

    for (int i = 0; i < rows.size(); i++) {
        Ctxt m = mult(rows[i], matrix);

        m = rotsum(m, 64, 1);

        columns.push_back(m);
    }

    return columns;
}

vector<Ctxt> FHEController::matmulCR(vector<Ctxt> rows, const Ptxt& weight, const Ptxt& bias) {
    vector<Ctxt> columns;

    for (int i = 0; i < rows.size(); i++) {
        Ctxt m = mult(rows[i], weight);

        m = rotsum(m, 128, 1);

        if (bias != nullptr) m = add(m, bias);

        columns.push_back(m);
    }

    return columns;
}

vector<Ctxt> FHEController::matmulCRlarge(vector<vector<Ctxt>> rows, vector<Ptxt> weights, const Ptxt &bias) {
    vector<Ctxt> output;

    for (int i = 0; i < rows.size(); i++) {
        //Qua sotto posso fare prima add-many e poi un solo rotsum mi sa:)
        /*
        Ctxt p1 = rotsum(mult(rows[i][0], weights[0]), 128, 1);
        Ctxt p2 = rotsum(mult(rows[i][1], weights[1]), 128, 1);
        Ctxt p3 = rotsum(mult(rows[i][2], weights[2]), 128, 1);
        Ctxt p4 = rotsum(mult(rows[i][3], weights[3]), 128, 1);

        Ctxt res = add({p1, p2, p3, p4});
        */

        Ctxt p1 = mult(rows[i][0], weights[0]);
        Ctxt p2 = mult(rows[i][1], weights[1]);
        Ctxt p3 = mult(rows[i][2], weights[2]);
        Ctxt p4 = mult(rows[i][3], weights[3]);

        Ctxt res = add({p1, p2, p3, p4});
        res = rotsum(res, 128, 1);

        if (bias != nullptr) res = add(res, bias);

        output.push_back(res);
    }

    return output;
}

Ctxt FHEController::matmulScores(vector<Ctxt> queries, const Ctxt &key) {
    vector<Ctxt> scores = matmulCR(queries, key);

    double r = 1 / 8.0; //Later corrected with e^(x/r)

    Ctxt scores_wrapped = mask_heads(scores[scores.size() - 1], 1 / 8.0 * r);
    scores_wrapped = rotate(scores_wrapped, -1);

    for (int i = scores.size() - 2; i >= 0; i--) {
        scores_wrapped = add(scores_wrapped,
                             mask_heads(scores[i], 1 / 8.0 * r));

        if (i > 0) scores_wrapped = rotate(scores_wrapped, -1);
    }

    return scores_wrapped;
}

Ctxt FHEController::wrapUpRepeated(vector<Ctxt> vectors) {
    vector<Ctxt> masked;

    for (int i = 0; i < vectors.size(); i++) {
        masked.push_back(mask_block(vectors[i], 128 * i, 128 * (i + 1), 1));
    }

    return context->EvalAddMany(masked);
}

Ctxt FHEController::wrapUpExpanded(vector<Ctxt> vectors) {
    //I use a vector to contain all the partial computations so that EvalAddTree adds less error than summing each
    //iteration
    Ctxt masked = mask_mod_n(vectors[vectors.size() - 1], 128);
    if (vectors.size() > 1) masked = rotate(masked, -1);

    for (int i = vectors.size() - 2; i >= 0; i--) {
        masked = add(masked, mask_mod_n(vectors[i], 128));
        if (i > 0) {
            masked = rotate(masked, -1);
        }
    }

    return masked;
}

vector<Ctxt> FHEController::unwrapExpanded(Ctxt c, int inputs_num) {
    vector<Ctxt> result;

    for (int i = 0; i < inputs_num; i++) {
        Ctxt out = mask_mod_n(c, 128, 0,inputs_num * 128);
        out = repeat(out, 128);


        if (i < inputs_num - 1) c = rotate(c, 1);

        result.push_back(out);
    }

    return result;
}

vector<vector<Ctxt>> FHEController::unwrapRepeatedLarge(vector<Ctxt> containers, int input_number) {
    vector<vector<Ctxt>> unwrapped_output;
    vector<int> quantities;

    for (int i = 0; i < input_number / 32.0; i++) {
        int quantity = 32;
        if ((i + 1) * 32 > input_number) {
            quantity = input_number - (i * 32);
        }

        quantities.push_back(quantity);
    }

    for (int i = 0; i < containers.size(); i++) {
        for (int j = 0; j < quantities[i]; j++) {
            vector<Ctxt> unwrapped_container = unwrap_512_in_4_128(containers[i], j);
            unwrapped_output.push_back(unwrapped_container);
        }
    }

    return unwrapped_output;
}

vector<Ctxt> FHEController::unwrapScoresExpanded(Ctxt c, int inputs_num) {
    vector<Ctxt> result;

    for (int i = 0; i < inputs_num; i++) {
        Ctxt i_th_1 = mask_mod_n(c, 128, 0,inputs_num * 128);
        Ctxt i_th_2 = mask_mod_n(c, 128, 64, inputs_num * 128);
        i_th_1 = repeat(i_th_1, 64);
        i_th_2 = repeat(i_th_2, 64);

        if (i < inputs_num - 1) c = rotate(c, 1);

        result.push_back(add(i_th_1, i_th_2));
    }

    return result;
}

vector<Ctxt> FHEController::unwrap_512_in_4_128(const Ctxt &c, int index) {
    vector<Ctxt> result;

    int shift = index * 512;

    Ctxt score1 = mask_block(c, shift + 0, shift + 128, 1);
    score1 = repeat(score1, 128, -128);
    Ctxt score2 = mask_block(c, shift + 128, shift + 256, 1);
    score2 = repeat(score2, 128, -128);
    Ctxt score3 = mask_block(c, shift+ 256, shift + 384, 1);
    score3 = repeat(score3, 128, -128);
    Ctxt score4 = mask_block(c, shift + 384, shift + 512, 1);
    score4 = repeat(score4, 128, -128);

    result.push_back(score1);
    result.push_back(score2);
    result.push_back(score3);
    result.push_back(score4);

    return result;
}

vector<Ctxt> FHEController::generate_containers(vector<Ctxt> inputs, const Ptxt& bias) {
    vector<Ctxt> containers;
    vector<int> quantities;

    reverse(inputs.begin(), inputs.end());

    for (int i = 0; i < inputs.size() / 32.0; i++) {
        int quantity = 32;
        if ((i + 1) * 32 > inputs.size()) {
            quantity = inputs.size() - (i * 32);
        }

        quantities.push_back(quantity);

        Ctxt partial_container = wrap_containers(slicing(inputs, (i) * 32, (i + 1) * 32), quantity);
        partial_container = add(partial_container, bias);
        containers.push_back(partial_container);
    }

    return containers;
}

Ctxt FHEController::wrap_containers(vector<Ctxt> c, int inputs_number) {
    //Resulting Ctxt will contain all the ciphertexts as follows:
    //c_n | c_n-1 | c_n-2 | ... | c_0
    Ctxt result = c[0];

    for (int i = 1; i < inputs_number; i++) {
        result = rotate(result, -512);
        result = add(result, c[i]);
    }

    return result;
}

Ctxt FHEController::mask_block(const Ctxt& c, int from, int to, double mask_value) {
    vector<double> mask;

    for (int i = 0; i < num_slots; i++) {
        if (i >= from && i < to) {
            mask.push_back(mask_value);
        } else {
            mask.push_back(0);
        }
    }

    return mult(c, encode(mask, c->GetLevel(), num_slots));
}

Ctxt FHEController::mask_heads(const Ctxt& c, double mask_value) {
    vector<double> mask;

    for (int i = 0; i < num_slots; i++) {
        if (i % 64 == 0) {
            mask.push_back(mask_value);
        } else {
            mask.push_back(0);
        }
    }

    return mult(c, encode(mask, c->GetLevel(), num_slots));
}

Ctxt FHEController::mask_mod_n(const Ctxt& c, int n) {
    vector<double> mask;
    for (int i = 0; i < num_slots; i++) {
        if (i % n == 0) {
            mask.push_back(1);
        } else {
            mask.push_back(0);
        }
    }

    return mult(c, encode(mask, c->GetLevel(), num_slots));
}

Ctxt FHEController::mask_mod_n(const Ctxt& c, int n, int padding, int max_slots) {
    vector<double> mask;
    for (int i = 0; i < num_slots; i++) {
        if (i % n == padding) {
            mask.push_back(1);
        } else {
            mask.push_back(0);
        }
    }

    return mult(c, encode(mask, c->GetLevel(), num_slots));
}

Ctxt FHEController::mask_first_n(const Ctxt &c, int n, double mask_value) {
    vector<double> mask;
    for (int i = 0; i < num_slots; i++) {
        if (i < n) {
            mask.push_back(mask_value);
        } else {
            mask.push_back(0);
        }
    }

    return mult(c, encode(mask, c->GetLevel(), num_slots));
}


Ctxt FHEController::eval_exp(const Ctxt &c, int inputs_number) {
    //Coefficients of Taylor series
    Ctxt res = context->EvalPoly(c, {1, 1, 1/(2.0), 1/(6.0), 1/(24.0), 1/(120.0), 1/(720.0)});

    if (res->GetLevel() + 4 > circuit_depth) {
        res = bootstrap(res);
    }

    res = context->EvalMultMany({res, res, res, res, res, res, res, res});

    //values must be corrected, slots that were 0 will now be 1, and this will break the following computations
    vector<double> mask;
    for (int i = 0; i < num_slots; i++) {
        //Here 12 è il numero di token, da cambiare
        if (i % 64 < inputs_number && i < (128 * inputs_number)) {
            mask.push_back(0);
        } else {
            mask.push_back(-1);
        }
    }

    return add(res, encode(mask, res->GetLevel(), num_slots));
}

Ctxt FHEController::eval_inverse(const Ctxt &c, double min, double max) {
    double middle = (max - min) / 2; //9995

    Ctxt res = add(c, encode(-middle - min, c->GetLevel(), num_slots)); // lo centro
    res = mult(res, encode(1 / middle, res->GetLevel(), num_slots)); //basta prima mascherare con 1 /9995 e addare -10005/9995 dopopì

    return context->EvalChebyshevFunction([](double x) -> double { return 1 / ((x * 9895) + 9995); }, res, -1, 1, 200);
}

Ctxt FHEController::eval_inverse_naive(const Ctxt &c, double min, double max) {
    return context->EvalChebyshevFunction([](double x) -> double { return 1 / x; }, c, min, max, 119);
}

Ctxt FHEController::eval_inverse_naive_2(const Ctxt &c, double min, double max, double mult) {
    return context->EvalChebyshevFunction([mult](double x) -> double { return mult / x; }, c, min, max, 200);
}

Ctxt FHEController::eval_gelu_function(const Ctxt &c, double min, double max, double mult, int degree) {
    return context->EvalChebyshevFunction([mult](double x) -> double { return  (0.5 * (x * (1 / mult)) * (1 + erf((x * (1 / mult)) / 1.41421356237))); }, c, min, max, degree);
}

Ctxt FHEController::eval_tanh_function(const Ctxt &c, double min, double max, double mult, int degree) {
    return context->EvalChebyshevFunction([mult](double x) -> double { return tanh(x * (1 / mult)); }, c, min, max, degree);
}

vector<Ctxt> FHEController::slicing(vector<Ctxt> &arr, int X, int Y) {
    if (Y - X >= arr.size())
        return arr;

    // Starting and Ending iterators
    auto start = arr.begin() + X;
    auto end = arr.begin() + Y;

    // To store the sliced vector
    vector<Ctxt> result(Y - X);

    // Copy vector using copy function()
    copy(start, end, result.begin());

    // Return the final sliced vector
    return result;
}


void FHEController::save(Ctxt v, std::string filename) {
    Serial::SerializeToFile(filename, v,
                            SerType::BINARY);
}

void FHEController::save(vector<Ctxt> v, std::string filename) {
    Serial::SerializeToFile(filename, v,
                            SerType::BINARY);
}

vector<Ctxt> FHEController::load_vector(string filename) {
    vector<Ctxt> result;

    if (!Serial::DeserializeFromFile(filename, result,
                                     SerType::BINARY)) {
        cerr << "Could not find \"" << filename << "\""
             << endl;

    }

    return result;
}

Ctxt FHEController::load_ciphertext(string filename) {
    Ctxt result;

    if (!Serial::DeserializeFromFile(filename, result,
                                     SerType::BINARY)) {
        cerr << "Could not find \"" << filename << "\""
             << endl;

    }

    return result;
}