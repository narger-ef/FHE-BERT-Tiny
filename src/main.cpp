#include <iostream>
#include "FHEController.h"
#include <chrono>
using namespace std::chrono;

enum class Parameters { Generate, Load };

FHEController controller;

vector<Ctxt> encoder1();
vector<Ctxt> encoder2(vector<Ctxt> input);

int main() {
    Parameters p = Parameters::Load;

    if (p == Parameters::Generate) {
        controller.generate_context(true);
        vector<int> rotations = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, -1, -2, -4, -8, -16, -32, -64};
        controller.generate_bootstrapping_and_rotation_keys(rotations, 16384, true, "rotation_keys.txt");
    } else if (p == Parameters::Load) {
        controller.load_context(false);
        controller.load_bootstrapping_and_rotation_keys("rotation_keys.txt", 16384, false);
    }

    cout << "The evaluation of the circuit started." << endl << endl;

    auto start = high_resolution_clock::now();

    //al 24 si bootstrappa

    vector<Ctxt> encoder1output, encoder2output;
    //Esce al 22 qua sotto

    encoder1output = encoder1();
    encoder1output = controller.load_vector("../checkpoint/encoder1output.bin");
    //Esce al 23 qua sotto

    encoder2output = encoder2(encoder1output);
    encoder2output = controller.load_vector("../checkpoint/encoder2output.bin");

    cout << "The whole thing took: " << (duration_cast<milliseconds>( high_resolution_clock::now() - start)).count() / 1000.0 << " seconds." << endl;



}

vector<Ctxt> encoder2(vector<Ctxt> inputs) {
    auto start = high_resolution_clock::now();

    Ptxt query_w = controller.read_plain_input("../weights/layer1_attself_query_weight.txt", inputs[0]->GetLevel());
    Ptxt query_b = controller.read_plain_repeated_input("../weights/layer1_attself_query_bias.txt", inputs[0]->GetLevel());
    Ptxt key_w = controller.read_plain_input("../weights/layer1_attself_key_weight.txt", inputs[0]->GetLevel());
    Ptxt key_b = controller.read_plain_repeated_input("../weights/layer1_attself_key_bias.txt", inputs[0]->GetLevel());

    vector<Ctxt> Q = controller.matmulRE(inputs, query_w, query_b);
    vector<Ctxt> K = controller.matmulRE(inputs, key_w, key_b);

    Ctxt K_wrapped = controller.wrapUpRepeated(K);

    Ctxt scores = controller.matmulScores(Q, K_wrapped);

    //scores = controller.bootstrap(scores);
    scores = controller.eval_exp(scores, inputs.size());

    scores = controller.mult(scores, 1 / 100.0);
    scores = controller.bootstrap(scores);
    scores = controller.mult(scores, 100.0);

    Ctxt scores_sum = controller.rotsum(scores, 128, 128);

    Ctxt scores_denominator = controller.eval_inverse_naive_2(scores_sum, 4, 200000, 1);

    scores_denominator = controller.bootstrap(scores_denominator);

    scores = controller.mult(scores, scores_denominator);

    vector<Ctxt> unwrapped_scores = controller.unwrapScoresExpanded(scores, inputs.size());

    Ptxt value_w = controller.read_plain_input("../weights/layer1_attself_value_weight.txt", inputs[0]->GetLevel());
    Ptxt value_b = controller.read_plain_repeated_input("../weights/layer1_attself_value_bias.txt", inputs[0]->GetLevel());

    vector<Ctxt> V = controller.matmulRE(inputs, value_w, value_b);

    Ctxt V_wrapped = controller.wrapUpRepeated(V);

    vector<Ctxt> output = controller.matmulRE(unwrapped_scores, V_wrapped, 128, 128);

    cout << "The evaluation of Self-Attention took: " << (duration_cast<milliseconds>( high_resolution_clock::now() - start)).count() / 1000.0 << " seconds." << endl;
    controller.print(controller.wrapUpRepeated(output), 128, "Self-Attention (Repeated)");
    //Qua la precisione è 0.9868

    /*
     * I remove all the ciphertexts except the first corresponding to the CLS token
     */
    Ctxt copyFirst = output[0];
    output.clear();
    output.push_back(copyFirst);

    start = high_resolution_clock::now();

    Ptxt dense_w = controller.read_plain_input("../weights/layer1_selfoutput_weight.txt", output[0]->GetLevel());
    Ptxt dense_b = controller.read_plain_expanded_input("../weights/layer1_selfoutput_bias.txt", output[0]->GetLevel() + 1); //Bias fai solo 12 ripetiz

    output = controller.matmulCR(output, dense_w, dense_b);

    for (int i = 0; i < output.size(); i++) {
        output[i] = controller.add(output[i], inputs[i]);
    }

    Ctxt wrappedOutput = controller.wrapUpExpanded(output);
    Ptxt precomputed_mean = controller.read_plain_repeated_input("../weights/layer1_selfoutput_mean.txt", wrappedOutput->GetLevel(), -1);
    wrappedOutput = controller.add(wrappedOutput, precomputed_mean);

    Ptxt vy = controller.read_plain_input("../weights/layer1_selfoutput_vy.txt", wrappedOutput->GetLevel(), 1);
    wrappedOutput = controller.mult(wrappedOutput, vy);
    Ptxt bias = controller.read_plain_expanded_input("../weights/layer1_selfoutput_normbias.txt", wrappedOutput->GetLevel(), 1, inputs.size());
    wrappedOutput = controller.add(wrappedOutput, bias);

    wrappedOutput = controller.bootstrap(wrappedOutput);

    Ctxt output_copy = wrappedOutput->Clone(); //Required at the last layernorm

    output = controller.unwrapExpanded(wrappedOutput, inputs.size());

    cout << "The evaluation of Self-Output took: " << (duration_cast<milliseconds>( high_resolution_clock::now() - start)).count() / 1000.0 << " seconds." << endl;
    controller.print_expanded(wrappedOutput, 0, 128, "Self-Output (Expanded)");
    //Fino a qui ottengo precisione 0.9828


    start = high_resolution_clock::now();

    double GELU_max_abs_value = 1 / 16.0;

    Ptxt intermediate_w_1 = controller.read_plain_input("../weights/layer1_intermediate_weight1.txt", wrappedOutput->GetLevel(), GELU_max_abs_value);
    Ptxt intermediate_w_2 = controller.read_plain_input("../weights/layer1_intermediate_weight2.txt", wrappedOutput->GetLevel(), GELU_max_abs_value);
    Ptxt intermediate_w_3 = controller.read_plain_input("../weights/layer1_intermediate_weight3.txt", wrappedOutput->GetLevel(), GELU_max_abs_value);
    Ptxt intermediate_w_4 = controller.read_plain_input("../weights/layer1_intermediate_weight4.txt", wrappedOutput->GetLevel(), GELU_max_abs_value);

    vector<Ptxt> dense_weights = {intermediate_w_1, intermediate_w_2, intermediate_w_3, intermediate_w_4};

    output = controller.matmulRElarge(output, dense_weights, nullptr);

    Ptxt intermediate_bias = controller.read_plain_repeated_512_input("../weights/layer1_intermediate_bias.txt", output[0]->GetLevel(), GELU_max_abs_value);

    output = controller.generate_containers(output, intermediate_bias);

    for (int i = 0; i < output.size(); i++) {
        output[i] = controller.eval_gelu_function(output[i], -1, 1, GELU_max_abs_value, 119);
        output[i] = controller.bootstrap(output[i]);
    }

    vector<vector<Ctxt>> unwrappedLargeOutput = controller.unwrapRepeatedLarge(output, output.size());

    cout << "The evaluation of Intermediate took: " << (duration_cast<milliseconds>( high_resolution_clock::now() - start)).count() / 1000.0 << " seconds." << endl;
    controller.print(unwrappedLargeOutput[0][0], 128, "Intermediate (Containers)");

    Ptxt output_w_1 = controller.read_plain_input("../weights/layer1_output_weight1.txt", output[0]->GetLevel());
    Ptxt output_w_2 = controller.read_plain_input("../weights/layer1_output_weight2.txt", output[0]->GetLevel());
    Ptxt output_w_3 = controller.read_plain_input("../weights/layer1_output_weight3.txt", output[0]->GetLevel());
    Ptxt output_w_4 = controller.read_plain_input("../weights/layer1_output_weight4.txt", output[0]->GetLevel());

    Ptxt output_bias = controller.read_plain_expanded_input("../weights/layer1_output_bias.txt", output[0]->GetLevel() + 1);

    output = controller.matmulCRlarge(unwrappedLargeOutput, {output_w_1, output_w_2, output_w_3, output_w_4}, output_bias);
    wrappedOutput = controller.wrapUpExpanded(output);

    wrappedOutput = controller.add(wrappedOutput, output_copy);

    precomputed_mean = controller.read_plain_repeated_input("../weights/layer1_output_mean.txt", wrappedOutput->GetLevel(), -1);
    wrappedOutput = controller.add(wrappedOutput, precomputed_mean);

    vy = controller.read_plain_input("../weights/layer1_output_vy.txt", wrappedOutput->GetLevel(), 1);
    wrappedOutput = controller.mult(wrappedOutput, vy);
    bias = controller.read_plain_expanded_input("../weights/layer1_output_normbias.txt", wrappedOutput->GetLevel(), 1, inputs.size());
    wrappedOutput = controller.add(wrappedOutput, bias);

    cout << "The evaluation of Output took: " << (duration_cast<milliseconds>( high_resolution_clock::now() - start)).count() / 1000.0 << " seconds." << endl;
    controller.print_expanded(wrappedOutput, 0, 128, "Output (Expanded)");

    //Precisione finale con approx: 0.9895
    //Precisione finale con reale : 0.8589

    controller.save(output, "../checkpoint/encoder2output.bin");

    return inputs;
}

vector<Ctxt> encoder1() {
    auto start = high_resolution_clock::now();

    vector<Ctxt> inputs;
    for (int i = 0; i < 12; i++) {
        inputs.push_back(controller.read_expanded_input("../input/input_" + to_string(i) + ".txt"));
    }

    Ptxt query_w = controller.read_plain_input("../weights/layer0_attself_query_weight.txt");
    Ptxt query_b = controller.read_plain_repeated_input("../weights/layer0_attself_query_bias.txt");
    Ptxt key_w = controller.read_plain_input("../weights/layer0_attself_key_weight.txt");
    Ptxt key_b = controller.read_plain_repeated_input("../weights/layer0_attself_key_bias.txt");

    vector<Ctxt> Q = controller.matmulRE(inputs, query_w, query_b);
    vector<Ctxt> K = controller.matmulRE(inputs, key_w, key_b);

    Ctxt K_wrapped = controller.wrapUpRepeated(K);

    Ctxt scores = controller.matmulScores(Q, K_wrapped);
    scores = controller.eval_exp(scores, inputs.size());

    Ctxt scores_sum = controller.rotsum(scores, 128, 128);
    Ctxt scores_denominator = controller.eval_inverse_naive(scores_sum, 2, 5000);

    scores = controller.mult(scores, scores_denominator);

    vector<Ctxt> unwrapped_scores = controller.unwrapScoresExpanded(scores, inputs.size());

    Ptxt value_w = controller.read_plain_input("../weights/layer0_attself_value_weight.txt", scores->GetLevel() - 2);
    Ptxt value_b = controller.read_plain_repeated_input("../weights/layer0_attself_value_bias.txt", scores->GetLevel() - 1);

    vector<Ctxt> V = controller.matmulRE(inputs, value_w, value_b);
    Ctxt V_wrapped = controller.wrapUpRepeated(V);

    vector<Ctxt> output = controller.matmulRE(unwrapped_scores, V_wrapped, 128, 128);

    cout << "The evaluation of Self-Attention took: " << (duration_cast<milliseconds>( high_resolution_clock::now() - start)).count() / 1000.0 << " seconds." << endl;
    controller.print(controller.wrapUpRepeated(output), 128, "Self-Attention (Repeated)");
    //Fino a qui ottengo precisione 0.9934

    start = high_resolution_clock::now();

    Ptxt dense_w = controller.read_plain_input("../weights/layer0_selfoutput_weight.txt", output[0]->GetLevel());
    Ptxt dense_b = controller.read_plain_expanded_input("../weights/layer0_selfoutput_bias.txt", output[0]->GetLevel() + 1); //Bias fai solo 12 ripetiz

    output = controller.matmulCR(output, dense_w, dense_b);

    for (int i = 0; i < output.size(); i++) {
        output[i] = controller.add(output[i], inputs[i]);
    }

    Ctxt wrappedOutput = controller.wrapUpExpanded(output);
    wrappedOutput = controller.bootstrap(wrappedOutput);

    Ptxt precomputed_mean = controller.read_plain_repeated_input("../weights/layer0_selfoutput_mean.txt", wrappedOutput->GetLevel(), -1);
    wrappedOutput = controller.add(wrappedOutput, precomputed_mean);

    Ptxt vy = controller.read_plain_input("../weights/layer0_selfoutput_vy.txt", wrappedOutput->GetLevel(), 1);
    wrappedOutput = controller.mult(wrappedOutput, vy);
    Ptxt bias = controller.read_plain_expanded_input("../weights/layer0_selfoutput_normbias.txt", wrappedOutput->GetLevel(), 1, inputs.size());
    wrappedOutput = controller.add(wrappedOutput, bias);

    Ctxt output_copy = wrappedOutput->Clone(); //Required at the last layernorm

    output = controller.unwrapExpanded(wrappedOutput, inputs.size());

    cout << "The evaluation of Self-Output took: " << (duration_cast<milliseconds>( high_resolution_clock::now() - start)).count() / 1000.0 << " seconds." << endl;
    controller.print_expanded(output[0], 0, 128, "Self-Output (Expanded)");
    //Fino a qui ottengo precisione 0.9964

    start = high_resolution_clock::now();

    double GELU_max_abs_value = 1 / 14.0;

    Ptxt intermediate_w_1 = controller.read_plain_input("../weights/layer0_intermediate_weight1.txt", wrappedOutput->GetLevel(), GELU_max_abs_value);
    Ptxt intermediate_w_2 = controller.read_plain_input("../weights/layer0_intermediate_weight2.txt", wrappedOutput->GetLevel(), GELU_max_abs_value);
    Ptxt intermediate_w_3 = controller.read_plain_input("../weights/layer0_intermediate_weight3.txt", wrappedOutput->GetLevel(), GELU_max_abs_value);
    Ptxt intermediate_w_4 = controller.read_plain_input("../weights/layer0_intermediate_weight4.txt", wrappedOutput->GetLevel(), GELU_max_abs_value);

    vector<Ptxt> dense_weights = {intermediate_w_1, intermediate_w_2, intermediate_w_3, intermediate_w_4};

    output = controller.matmulRElarge(output, dense_weights, nullptr);

    Ptxt intermediate_bias = controller.read_plain_repeated_512_input("../weights/layer0_intermediate_bias.txt", output[0]->GetLevel(), GELU_max_abs_value);

    output = controller.generate_containers(output, intermediate_bias);

    for (int i = 0; i < output.size(); i++) {
        output[i] = controller.eval_gelu_function(output[i], -1, 1, GELU_max_abs_value, 59);
        output[i] = controller.bootstrap(output[i]);
    }

    vector<vector<Ctxt>> unwrappedLargeOutput = controller.unwrapRepeatedLarge(output, inputs.size());

    cout << "The evaluation of Intermediate took: " << (duration_cast<milliseconds>( high_resolution_clock::now() - start)).count() / 1000.0 << " seconds." << endl;
    controller.print(unwrappedLargeOutput[0][0], 128, "Intermediate (Containers)");
    //Fino a qui ottengo precisione 0.9957

    Ptxt output_w_1 = controller.read_plain_input("../weights/layer0_output_weight1.txt", unwrappedLargeOutput[0][0]->GetLevel());
    Ptxt output_w_2 = controller.read_plain_input("../weights/layer0_output_weight2.txt", unwrappedLargeOutput[0][0]->GetLevel());
    Ptxt output_w_3 = controller.read_plain_input("../weights/layer0_output_weight3.txt", unwrappedLargeOutput[0][0]->GetLevel());
    Ptxt output_w_4 = controller.read_plain_input("../weights/layer0_output_weight4.txt", unwrappedLargeOutput[0][0]->GetLevel());

    Ptxt output_bias = controller.read_plain_expanded_input("../weights/layer0_output_bias.txt", unwrappedLargeOutput[0][0]->GetLevel() + 1);

    output = controller.matmulCRlarge(unwrappedLargeOutput, {output_w_1, output_w_2, output_w_3, output_w_4}, output_bias);
    wrappedOutput = controller.wrapUpExpanded(output);

    wrappedOutput = controller.add(wrappedOutput, output_copy);

    precomputed_mean = controller.read_plain_repeated_input("../weights/layer0_output_mean.txt", wrappedOutput->GetLevel(), -1);
    wrappedOutput = controller.add(wrappedOutput, precomputed_mean);

    vy = controller.read_plain_input("../weights/layer0_output_vy.txt", wrappedOutput->GetLevel(), 1);
    wrappedOutput = controller.mult(wrappedOutput, vy);
    bias = controller.read_plain_expanded_input("../weights/layer0_output_normbias.txt", wrappedOutput->GetLevel(), 1, inputs.size());
    wrappedOutput = controller.add(wrappedOutput, bias);

    wrappedOutput = controller.bootstrap(wrappedOutput);

    output = controller.unwrapExpanded(wrappedOutput, inputs.size());

    cout << "The evaluation of Output took: " << (duration_cast<milliseconds>( high_resolution_clock::now() - start)).count() / 1000.0 << " seconds." << endl;
    controller.print(wrappedOutput, 0, "Output (Expanded)");
    //Fino a qui ottengo precisione 0.9965

    controller.save(output, "../checkpoint/encoder1output.bin");

    return output;
}


