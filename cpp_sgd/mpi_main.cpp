#include <unordered_map>
#include <string>
#include <iostream>
#include <fstream>      // std::ofstream
#include <sstream>
#include <memory>
#include <chrono>       // time measurements
#include <ctime>
#include <random>
#include <iterator>

// uncomment to disable assert()
// #define NDEBUG
#include <cassert>

// 3rd party libraries (eigen and OpenMPI)
#include <Eigen/Dense>
#include <mpi.h>

#include "sgd_types.h"
#include "ioutil.h"
#include "evaluation.h"
#include "sgd.h"
#include "variant.cpp"

// Define "tags" which identify messages sent between master and slave in  MPI
#define LENGTH_TAG 0
#define CONFIG_NAME_TAG 1
#define CONFIG_VALUE_TAG 2
#define SCORE_TAG 3
#define NEW_TASK_TAG 4
#define DIE_TAG 5

#define DEFAULT_BAD_SCORE 10000
#define WRITE_CONFIG_EVERY 50
#define MAX_TASKS 10000
#define CV_TEST_PERCENT 0.1
#define CV_K 3

using reccommend::dtype;
using reccommend::MatrixI;
using reccommend::MatrixD;
using reccommend::Settings;
using reccommend::SettingsEntry;
using reccommend::DataPair;
using reccommend::IOUtil;

// Variant type. Defines the possibilities for the parameter search 
// (either min/max, a list of values, or a single scalar)
using config_var = variant<std::pair<dtype, dtype>, // Ranges of floats and ints
                           std::pair<int, int>, 
                           std::vector<dtype>,      // Categorical variable (only floats for now)
                           dtype, int>;             // Single value.
// Variant type. Defines possible data types for configuration values.
using config_val = variant<dtype, int>;

typedef std::unordered_map<std::string, config_var> SettingsRange;


const static int NUSERS = 10000;
const static int NITEMS = 1000;
const static std::string TRAIN_DATA_FILE = "../data_train.csv";
const static std::string MASK_DATA_FILE = "../sampleSubmission.csv";


/**
 * Choose a random value within the allowed range (or list of allowed values)
 * from the supplied configuration parameter.
 */
config_val chooseParamValue(config_var param, std::mt19937 &rng);

/**
 * Choose a random parameter from a list of possible parameters.
 */
std::string chooseParamName(const SettingsRange &min_max_config, std::mt19937 &rng);

/**
 * Randomly choose one parameter, and the random value for that parameter
 * from the SettingsRange data.
 * Returns one configuration entry.
 */
SettingsEntry nextConfig(SettingsRange &min_max_config, std::mt19937 &rng);

/**
 * Generate the starting point for the parameter search with the Integrated solver.
 * Most parameters can be modified within a large range.
 */
SettingsRange getIntegratedConfig (const int num_threads);

/**
 * Configuration for searching parameters with the SVD solver.
 * Here large importance is given to num_factors, K1 and K2.
 */
SettingsRange getSVDConfig (const int num_threads);

/**
 * Execute the master routine.
 */
void master (SettingsRange &min_max_config,
             unsigned long rseed,
             uint          max_tasks,
             std::string   config_save_file);

/**
 * Execute the slave routine.
 *
 * Template parameter tells which kind of solver (from sgd.h) will be used.
 */
template <class Solver>
void slave (ulong data_rseed);

/**
 * Helper for obtaining the list of parameter names from a specific configuration.
 */
std::vector<std::string> configToNameVector (const Settings &c);

/**
 * Helper for obtaining the list of parameter values from a specific configuration.
 */
std::vector<dtype> configToValVector (const Settings &c);

/**
 * Implements the protocol for sending, and symmetrically for receiving
 * a specific configuration to/from the process with the specified rank.
 */
void sendConfig(const Settings &c, int rank);
Settings recvConfig(int rank);

/**
 * Write a list of configurations, appending to specified file. (TODO: should be in IOUtil.cpp)
 */
void write_config_to_csv(std::vector<std::pair<Settings, double>>::iterator cbegin,
                         std::vector<std::pair<Settings, double>>::iterator cend,
                         std::string fname);


config_val chooseParamValue(config_var param, std::mt19937 &rng)
{
    config_val res;
    res.set<dtype>(0); // default (unneeded)
    if (param.is<pair<dtype, dtype> >()) {
        std::uniform_real_distribution<dtype> param_dis (param.get<pair<dtype, dtype> >().first, 
                                                         param.get<pair<dtype, dtype> >().second);
        res.set<dtype>(param_dis(rng));
    }
    else if (param.is<pair<int, int> >()) {
        std::uniform_int_distribution<int> param_dis (param.get<pair<int, int> >().first, 
                                                       param.get<pair<int, int> >().second);
        res.set<int>(param_dis(rng));
    }
    else if (param.is<std::vector<dtype> >()) {
        std::uniform_real_distribution<dtype> param_dis (0, param.get<vector <dtype> >().size() - 1);
        res.set<dtype>(param.get<vector <dtype> >()[param_dis(rng)]);
    }
    else if (param.is<dtype>()) {
        res.set<dtype>(param.get<dtype>());
    }
    else if (param.is<int>()) {
        res.set<int>(param.get<int>());
    }
    return res;
}

std::string chooseParamName(
    const SettingsRange &min_max_config,
    std::mt19937 &rng)
{
    static std::uniform_int_distribution<int> uni(0, min_max_config.size() - 1); // guaranteed unbiased
    // Pick a random element in the map
    auto map_iter = min_max_config.begin();
    std::advance(map_iter, uni(rng));
    return map_iter->first;
}


SettingsRange getSVDConfig (const int num_threads) {
    SettingsRange min_max_config;
    config_var nusers_choice;
    nusers_choice.set<int>(NUSERS);
    min_max_config["nusers"] = nusers_choice;

    config_var nitems_choice;
    nitems_choice.set<int>(NITEMS);
    min_max_config["nitems"] = nitems_choice;

    config_var num_factors_choice;
    num_factors_choice.set<pair<int, int> >(pair<int, int>(2, 500));
    min_max_config["num_factors"] = num_factors_choice;

    config_var K1_choice;
    K1_choice.set<pair<int, int> >(pair<int, int>(1, 400));
    min_max_config["K1"] = K1_choice;

    config_var K2_choice;
    K2_choice.set<pair<int, int> >(pair<int, int>(1, 400));
    min_max_config["K2"] = K2_choice;

    config_var num_threads_choice;
    num_threads_choice.set<int>(1);
    min_max_config["num_threads"] = num_threads_choice;

    config_var max_iter_choice;
    max_iter_choice.set<int>(1);
    min_max_config["max_iter"] = max_iter_choice;

    return min_max_config;
}

SettingsRange getIntegratedConfig (const int num_threads) {
    /* Define the settings */
    SettingsRange min_max_config;
    config_var nusers_choice;
    nusers_choice.set<int>(NUSERS);
    min_max_config["nusers"] = nusers_choice;

    config_var nitems_choice;
    nitems_choice.set<int>(NITEMS);
    min_max_config["nitems"] = nitems_choice;

    config_var lrate1_choice;
    lrate1_choice.set<pair<dtype, dtype> >(pair<dtype, dtype>(0.001, 0.03));
    min_max_config["lrate1"] = lrate1_choice;

    config_var lrate2_choice;
    lrate2_choice.set<pair<dtype, dtype> >(pair<dtype, dtype>(0.001, 0.03));
    min_max_config["lrate2"] = lrate2_choice;
    
    config_var lrate3_choice;
    lrate3_choice.set<pair<dtype, dtype> >(pair<dtype, dtype>(0.001, 0.03));
    min_max_config["lrate3"] = lrate3_choice;
    
    config_var regl4_choice;
    regl4_choice.set<pair<dtype, dtype> >(pair<dtype, dtype>(0.001, 0.3));
    min_max_config["regl4"] = regl4_choice;
    
    config_var regl6_choice;
    regl6_choice.set<pair<dtype, dtype> >(pair<dtype, dtype>(0.001, 0.3));
    min_max_config["regl6"] = regl6_choice;
    
    config_var regl7_choice;
    regl7_choice.set<pair<dtype, dtype> >(pair<dtype, dtype>(0.001, 0.3));
    min_max_config["regl7"] = regl7_choice;
    
    config_var regl8_choice;
    regl8_choice.set<pair<dtype, dtype> >(pair<dtype, dtype>(0.001, 0.3));
    min_max_config["regl8"] = regl8_choice;
    
    config_var lrate_reduction_choice;
    lrate_reduction_choice.set<pair<dtype, dtype> >(pair<dtype, dtype>(0.9, 0.999));
    min_max_config["lrate_reduction"] = lrate_reduction_choice;
    
    config_var num_factors_choice;
    num_factors_choice.set<pair<int, int> >(pair<int, int>(10, 100));
    min_max_config["num_factors"] = num_factors_choice;
    
    config_var max_iter_choice;
    max_iter_choice.set<pair<int, int> >(pair<int, int>(10, 80));
    //max_iter_choice.set<int>(1);//30);
    min_max_config["max_iter"] = max_iter_choice;

    config_var correlation_shrinkage_choice;
    correlation_shrinkage_choice.set<int>(100);
    min_max_config["correlation_shrinkage"] = correlation_shrinkage_choice;
    
    config_var K1_choice;
    K1_choice.set<int>(1);
    min_max_config["K1"] = K1_choice;
    
    config_var K2_choice;
    K2_choice.set<int>(10);
    min_max_config["K2"] = K2_choice;
    
    config_var max_neigh_choice;
    max_neigh_choice.set<pair<int, int> >(pair<int, int>(20, 500));
    min_max_config["max_neigh"] = max_neigh_choice;
    
    config_var num_threads_choice;
    num_threads_choice.set<int>(num_threads);
    min_max_config["num_threads"] = num_threads_choice;

    return min_max_config;
}


SettingsEntry nextConfig(SettingsRange &min_max_config, std::mt19937 &rng)
{
    std::string param;
    dtype param_value;

    while (true) {
        param = chooseParamName(min_max_config, rng);
        if (min_max_config[param].is<dtype>() || min_max_config[param].is<int>()) {
            continue;
        }
        config_val param_val_var = chooseParamValue(min_max_config.at(param), rng);
        if (param_val_var.is<dtype>())
            param_value = param_val_var.get<dtype>();
        else
            param_value = static_cast<dtype>(param_val_var.get<int>());
        break;
    }
    return SettingsEntry(param, param_value);
}

std::vector<std::string> configToNameVector (const Settings &c)
{
    std::vector<std::string> nv(c.size());
    int i = 0;
    for ( auto it = c.begin(); it!= c.end(); ++it, ++i ) {
        nv[i] = it->first;
    }
    return nv;
}

std::vector<dtype> configToValVector (const Settings &c)
{
    std::vector<dtype> nv(c.size());
    int i = 0;
    for ( auto it = c.begin(); it!= c.end(); ++it, ++i ) {
        nv[i] = it->second;
    }
    return nv;
}

void sendConfig(const Settings &c, int rank)
{
    auto names = configToNameVector (c);
    auto values = configToValVector (c);
    // Send length, then names, then values
    int len = c.size();
    MPI_Send( &len,           1,   MPI_INT,    rank, LENGTH_TAG,       MPI_COMM_WORLD );
    for (int i = 0; i < len; i++) {
        int nlen = names[i].size();
        const char *name = names[i].data();
        MPI_Send( &nlen,      1,    MPI_INT,   rank, LENGTH_TAG,       MPI_COMM_WORLD);
        MPI_Send( name,       nlen, MPI_CHAR,  rank, CONFIG_NAME_TAG,  MPI_COMM_WORLD );
        MPI_Send( &values[i], 1,    MPI_FLOAT, rank, CONFIG_VALUE_TAG, MPI_COMM_WORLD );   
    }
}

Settings recvConfig(int rank)
{
    MPI_Status status;
    int len, nlen;
    dtype val;
    Settings c;
    MPI_Recv( &len, 1, MPI_INT, rank, LENGTH_TAG, MPI_COMM_WORLD, &status);
    for (int i = 0; i < len; i++) {
        MPI_Recv( &nlen, 1,    MPI_INT,    rank, LENGTH_TAG,       MPI_COMM_WORLD, &status);
        char *name = new char[nlen];
        MPI_Recv( name,  nlen, MPI_CHAR,   rank, CONFIG_NAME_TAG,  MPI_COMM_WORLD, &status);
        MPI_Recv( &val,   1,    MPI_FLOAT,  rank, CONFIG_VALUE_TAG, MPI_COMM_WORLD, &status);
        c[std::string(name, nlen)] = val;
        delete[] name;
    }
    return c;
}



int main(int argc, char** argv) 
{
    MPI_Init(&argc, &argv);
 
    // Force flushing of output
    std::cout.setf( std::ios_base::unitbuf );

    unsigned long rseed = 12351;
    unsigned long data_rseed = 1633419;
    int           num_threads = 1;
    int           rank;
    std::string   csave = "../saved_data/svd_rsearch_configs.csv";

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Read command line argument */
    std::string searchType;
    if (argc > 1) {
        searchType = argv[1];
    } else {
        std::cout << "No argument found for 'search type'. Please specify a command line argument:\n"
                  << "MPI_SGD <integrated>|<svd>\n";
        return 1;
    }


    if (rank == 0) { /* Master routine */
        SettingsRange min_max_config;
        if (searchType == "integrated") {
            min_max_config = getIntegratedConfig(num_threads);
        }
        else if (searchType == "svd") {
            min_max_config = getSVDConfig(num_threads);
        }
        else {
            std::cout << "Chosen search type (" << searchType << ") is not valid." << "\n";
            return 2;
        }
        master (min_max_config, rseed, MAX_TASKS, csave);
    }
    else { /* Slave routine */
        if (searchType == "integrated") {
            slave<reccommend::IntegratedSolver> (data_rseed);
        }
        else if (searchType == "svd") {
            slave<reccommend::SVD> (data_rseed);
        }
        else {
            std::cout << "Chosen search type (" << searchType << ") is not valid." << "\n";
            return 2;
        }
    }
    MPI_Finalize();
    return 0;
}



void write_config_to_csv(std::vector<std::pair<Settings, double>>::iterator cbegin, 
    std::vector<std::pair<Settings, double>>::iterator cend,
    std::string fname) 
{
    std::ofstream ofs;
    ofs.open (fname, std::ofstream::out | std::ofstream::app);
    
    for (; cbegin != cend; ++cbegin) {
        Settings config = cbegin->first;
        double score = cbegin->second;
        ofs << score << ",";
        for (auto it = config.begin(); it != config.end(); ++it) {
            ofs << it->first << ":" << it->second << ",";
        }
        ofs << "\n";
    }

    ofs.close();
}


typedef std::chrono::time_point<std::chrono::high_resolution_clock> dTimePoint;

void master (SettingsRange &min_max_config, 
    unsigned long rseed,
    uint max_tasks,
    std::string config_save_file)
{
    MPI_Status     status;
    int            ntasks;
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    uint           sent_tasks = 1;
    std::vector<std::pair<Settings, double>> config_scores;
    double         best_score = DEFAULT_BAD_SCORE;
    std::unordered_map<int, dTimePoint> task_start;
    
    /*
     * Generate initial configuration
     */
    std::mt19937 rng(rseed);

    std::cout << "Initial configuration \n";
    Settings curr_config;
    for (auto it = min_max_config.begin(); it != min_max_config.end(); it++) {
        auto param_val_var = chooseParamValue(it->second, rng); 
        if (param_val_var.is<dtype>())
            curr_config[it->first] = param_val_var.get<dtype>();
        else
            curr_config[it->first] = static_cast<dtype>(param_val_var.get<int>());
        std::cout << "\t" << it->first << " : " << curr_config[it->first] << std::endl;
    }
    std::cout << "\n";

    /*
     * Seed the slaves (1 config per slave)
     */
    for (int rank = 1; rank < ntasks; rank++) {
        auto nparam = nextConfig(min_max_config, rng);
        curr_config[nparam.first] = nparam.second;
        MPI_Send(0, 0, MPI_INT, rank, NEW_TASK_TAG, MPI_COMM_WORLD);
        sendConfig(curr_config, rank);

        task_start[rank] = std::chrono::high_resolution_clock::now();
        std::cout << reccommend::now() << "Sent task " << sent_tasks << " to node " << rank << "\n";
        sent_tasks++;
    }

    /*
     * Receive a result from any slave;
     * Dispatch a new config to the slaves which are finished.
     */
    while (sent_tasks <= max_tasks) {
        if (config_scores.size() > 0 && config_scores.size() % WRITE_CONFIG_EVERY == 0) {
            write_config_to_csv(config_scores.end()-WRITE_CONFIG_EVERY,
                config_scores.end(),
                config_save_file);
            std::cout << reccommend::now() << config_scores.size() << " results written to " << config_save_file << "\n";
        }
        double test_score;
        MPI_Recv( &test_score, 1, MPI_DOUBLE, MPI_ANY_SOURCE, SCORE_TAG, MPI_COMM_WORLD, &status);
        int source = status.MPI_SOURCE;
        auto rconfig = recvConfig(source);

        // Report & Store obtained results.
        if (test_score != test_score) {
            test_score = DEFAULT_BAD_SCORE;
        }
        std::cout << reccommend::now() << "Received task from " << source
                  << ". Score = " << test_score 
                  << ", Elapsed = " << static_cast<int>(reccommend::elapsed(task_start[source]) / 1000) << "s\n";

        auto copied_config = rconfig;
        config_scores.push_back(
            std::pair<Settings, double>(copied_config, test_score));
        if (test_score < best_score) {
            best_score = test_score;
            std::cout << "Best configuration up to now: \n";
            for (auto it = rconfig.begin(); it != rconfig.end(); it++) {
                std::cout << "\t" << it->first << " : " << it->second << "\n";
            }
            std::cout << "\n";
        }

        // Send new task
        auto nparam = nextConfig(min_max_config, rng);
        curr_config[nparam.first] = nparam.second;

        MPI_Send(0, 0, MPI_INT, source, NEW_TASK_TAG, MPI_COMM_WORLD);
        sendConfig(curr_config, source);

        task_start[source] = std::chrono::high_resolution_clock::now();
        std::cout << reccommend::now() << "Sent task " << sent_tasks << " to " << source << "\n";
        sent_tasks++;
    }

    /*
     * Kill all other processes
     */
    for (int rank = 1; rank < ntasks; rank++) {
        MPI_Send(0, 0, MPI_INT, rank, DIE_TAG, MPI_COMM_WORLD);
    }
}


template <class Solver>
void slave (ulong data_rseed)
{
    MPI_Status status;
    int        len;
    double     test_score;

    // Read Data
    auto cv_data = IOUtil::readDataCV(TRAIN_DATA_FILE, CV_K, NUSERS, NITEMS, CV_TEST_PERCENT, data_rseed);

    while (true) {
        /* Read messages in */
        MPI_Recv( 0, 0, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        if (status.MPI_TAG == DIE_TAG)
            return;
        Settings rconfig = recvConfig(0);

        /* Perform task */
        test_score = reccommend::kfoldCV<Solver>(CV_K, rconfig, cv_data, 0);

        /* Send messages out */
        MPI_Send( &test_score, 1,   MPI_DOUBLE, 0, SCORE_TAG,        MPI_COMM_WORLD );
        sendConfig(rconfig, 0);
    }
}
