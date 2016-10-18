/*

Copyright (c) 2005-2016, University of Oxford.
All rights reserved.

University of Oxford means the Chancellor, Masters and Scholars of the
University of Oxford, having an administrative office at Wellington
Square, Oxford OX1 2JD, UK.

This file is part of Chaste.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
 * Neither the name of the University of Oxford nor the names of its
   contributors may be used to endorse or promote products derived from this
   software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#ifndef TESTCRUMBPREDICTIONS_HPP_
#define TESTCRUMBPREDICTIONS_HPP_

// Includes first
#include <cxxtest/TestSuite.h>
#include <boost/assign.hpp>

// Standard Chaste classes
#include "FileFinder.hpp"

// ApPredict bolt-on project classes
#include "SingleActionPotentialPrediction.hpp"
#include "SetupModel.hpp"

// This project classes.
#include "CrumbDataReader.hpp"
#include "CrumbDrugList.hpp"

// To specify hierarchical or not
#include <CommandLineArguments.hpp>

// Should always be last
#include "PetscSetupAndFinalize.hpp"

class TestCrumbPredictions : public CxxTest::TestSuite
{
public:
    void TestCrumbDrugListReader(void) throw(Exception)
    {
        FileFinder file("projects/GaryM/test/private_data/crumb_study/drug_list.txt", RelativeTo::ChasteSourceRoot);

        if (!file.Exists())
        {
            EXCEPTION("Data files are missing.");
        }

        CrumbDrugList drug_list(file);
        TS_ASSERT_EQUALS(drug_list.GetNumDrugs(), 30u);

        TS_ASSERT_EQUALS(drug_list.GetDrugName(0u), "Amiodarone");
        TS_ASSERT_EQUALS(drug_list.GetDrugName(18u), "Propafenone");
        TS_ASSERT_EQUALS(drug_list.GetDrugName(29u), "Verapamil");

        TS_ASSERT_EQUALS(drug_list.GetRiskCategory("Dofetilide") , HIGH);
        TS_ASSERT_EQUALS(drug_list.GetRiskCategory("Terfenadine"), INTERMEDIATE);
        TS_ASSERT_EQUALS(drug_list.GetRiskCategory("Ranolazine") , LOW);

        TS_ASSERT_DELTA(drug_list.GetClinicalDose("Dofetilide") , 0.0021, 1e-9);
        TS_ASSERT_DELTA(drug_list.GetClinicalDose("Terfenadine"), 0.0003, 1e-9);
        TS_ASSERT_DELTA(drug_list.GetClinicalDose("Ranolazine") , 1.9482, 1e-9);
    }

    void TestCrumbInferredDoseResponseSamplesReader(void) throw(Exception)
    {
        unsigned num_params = 2u;
        std::string drug_name = "Amiodarone";
        std::stringstream file_path;
        file_path << "projects/GaryM/test/private_data/crumb_study/standard_mcmc/num_params_"
                  << num_params << "/" << drug_name << "_hERG_num_params_" << num_params << "_hill_pic50.txt";
        FileFinder drug_file_name(file_path.str(), RelativeTo::ChasteSourceRoot);
        CrumbDataReader data_reader(drug_file_name, num_params);

        TS_ASSERT_THROWS_THIS(data_reader.GetHillSample(1e9),
                              "Sample index 1000000000 requested, but the number of samples in the data file is 500.");
        TS_ASSERT_THROWS_THIS(data_reader.GetPic50Sample(1e9),
                              "Sample index 1000000000 requested, but the number of samples in the data file is 500.");

        TS_ASSERT_DELTA(data_reader.GetPic50Sample(0u),5.836964934087349732e+00, 1e-12);
        TS_ASSERT_DELTA(data_reader.GetPic50Sample(1u),5.862863573409464735e+00, 1e-12);
        TS_ASSERT_DELTA(data_reader.GetPic50Sample(499u),6.134491843253357501e+00, 1e-12);

        TS_ASSERT_DELTA(data_reader.GetHillSample(0u),5.707994444688915259e-01, 1e-12);
        TS_ASSERT_DELTA(data_reader.GetHillSample(1u),4.787830248016846646e-01, 1e-12);
        TS_ASSERT_DELTA(data_reader.GetHillSample(499u),5.957891765182190547e-01, 1e-12);
    }

    void TestApdPredictionsForCrumbData() throw (Exception)
    {
        const unsigned num_samples = 500u;
        //const unsigned num_samples = 10u;
        unsigned num_params = 2u;
        bool only_herg_cav = false;
        bool hierarchical = CommandLineArguments::Instance()->OptionExists("--hierarchical");



        // Load up the drug data
        FileFinder file("projects/GaryM/test/private_data/crumb_study/drug_list.txt", RelativeTo::ChasteSourceRoot);
        if (!file.Exists())
        {
            std::cout << "Data file not available." << std::endl;
            return;
        }
        CrumbDrugList drug_list(file);

        // Channel names in the Crumb paper and derived data files
        std::vector<std::string> channels = boost::assign::list_of("hERG")
        ("Nav1.5-peak")("Nav1.5-late")("Cav1.2")("KvLQT1_mink")("Kv4.3")("Kir2.1");
        // Corresponding channel names for the ApPredict arguments
        std::vector<std::string> ap_predict_channels = boost::assign::list_of
                ("membrane_rapid_delayed_rectifier_potassium_current_conductance")
                ("membrane_fast_sodium_current_conductance")
                ("membrane_persistent_sodium_current_conductance")
                ("membrane_L_type_calcium_current_conductance")
                ("membrane_slow_delayed_rectifier_potassium_current_conductance")
                ("membrane_fast_transient_outward_current_conductance")
                ("membrane_inward_rectifier_potassium_current_conductance");

        std::vector<std::string> subset_drugs;
        if (CommandLineArguments::Instance()->OptionExists("--drugs"))
        {
            subset_drugs = CommandLineArguments::Instance()->GetStringsCorrespondingToOption("--drugs");
        }

        std::stringstream output_folder;
        output_folder << "CrumbDataStudy_model_";

        unsigned model_idx = 6u;
        if (CommandLineArguments::Instance()->OptionExists("--cellml"))
        {
            std::string model_name = CommandLineArguments::Instance()->GetStringCorrespondingToOption("--cellml");
            FileFinder finder(model_name, RelativeTo::CWD);
            output_folder << finder.GetLeafNameNoExtension();
        }
        else
        {
            if (CommandLineArguments::Instance()->OptionExists("--model"))
            {
                model_idx = CommandLineArguments::Instance()->GetUnsignedCorrespondingToOption("--model");
            }
            output_folder << model_idx;
        }

        output_folder << "_num_params_" << num_params << "_num_samples_" << num_samples;

        if(only_herg_cav)
        {
            output_folder << "_hERG_and_CaL_only";
        }

        if (hierarchical)
        {
            output_folder << "_hierarchical";
        }


        // Set up an output directory - must be called collectively.
        boost::shared_ptr<OutputFileHandler> p_base_handler(new OutputFileHandler(output_folder.str(), false)); // Don't wipe the folder, we might be re-running one drug!

        PetscTools::Barrier("Output folder created"); // Shouldn't be needed but seemed to avoid an error!

        std::cout << "\n\nOutput folder created, apparently.\n\n";


        // Everyone does their own thing now...
        PetscTools::IsolateProcesses(true);

        // They all need their own working directory to do CellML conversion in...
        output_folder << "/" << PetscTools::GetMyRank();

        std::cout << "\n\noutput_folder: " << output_folder << "\n\n";


        // Loop over each compound
        for (unsigned drug_idx = 0; drug_idx<drug_list.GetNumDrugs(); drug_idx++)
        {
            const std::string drug_name = drug_list.GetDrugName(drug_idx);
            double concentration = drug_list.GetClinicalDose(drug_name);

            // If we are just doing a subset, and this drug isn't in the list, then skip it.
            if (subset_drugs.size()>0u && std::find(subset_drugs.begin(), subset_drugs.end(), drug_name) == subset_drugs.end())
            {
                // Don't examine this compound
                continue;
            }

            // If we are running in parallel share out the drugs between processes.
            if (drug_idx % PetscTools::GetNumProcs() != PetscTools::GetMyRank())
            {
                // Let another processor do this drug
                continue;
            }

            boost::shared_ptr<AbstractCvodeCell> p_model;

            if (CommandLineArguments::Instance()->OptionExists("--model") || CommandLineArguments::Instance()->OptionExists("--cellml"))
            {
                // Create a fresh model for this drug - SetupModel contains command line arg processing for which.
                boost::shared_ptr<OutputFileHandler> p_individual_handler(new OutputFileHandler(output_folder.str(), true));
                SetupModel setup(1.0,UNSIGNED_UNSET,p_individual_handler);
                p_model = setup.GetModel();
            }
            else
            {
                SetupModel setup(1.0, 6u); // Use an O'Hara model
                p_model = setup.GetModel();
            }

            // Run to steady state and record state variables
            SteadyStateRunner steady_runner(p_model);
            steady_runner.RunToSteadyState();
            N_Vector steady_state_variables = p_model->GetStateVariables();

            // Not all of the models have a distinct fast I_to component.
            // In this case we look for the complete I_to current instead.
            if (!p_model->HasParameter("membrane_fast_transient_outward_current_conductance") &&
                 p_model->HasParameter("membrane_transient_outward_current_conductance") )
            {
                WARNING(p_model->GetSystemName() << " does not have 'membrane_fast_transient_outward_current_conductance' labelled, using combined Ito (fast and slow) instead...");
                ap_predict_channels[5u] = "membrane_transient_outward_current_conductance";
            }

            // Record the default conductances for scaling purposes.
            c_vector<double, 7u> default_conductances;
            for (unsigned channel_idx = 0u; channel_idx<channels.size(); channel_idx++)
            {
                if (p_model->HasParameter(ap_predict_channels[channel_idx]))
                {
                    default_conductances[channel_idx] = p_model->GetParameter(ap_predict_channels[channel_idx]);
                }
                else
                {
                    WARN_ONCE_ONLY("Model " << p_model->GetSystemName() << " doesn't have '" << ap_predict_channels[channel_idx] << "' labelled, simulations ran without blocking it.");
                }
            }

            // Work out the name and open a log file
            std::cout << "COMPOUND = " << drug_name << std::endl;
            std::stringstream output_name;
            output_name << drug_name << "_apd90_results_num_params_" << num_params << ".dat";
            out_stream p_file = p_base_handler->OpenOutputFile(output_name.str());
            *p_file << "Sample\tConc(uM)\tAPD90(ms)\tAPD50(ms)\tCaMax(mM)\tCaMin(mM)" << std::endl;

            // Read in all the data for this compound at once
            std::vector<CrumbDataReader> data_readers; // One for each channel...
            for (unsigned channel_idx = 0; channel_idx < channels.size(); channel_idx++)
            {
                std::stringstream file_path;
                if (hierarchical)
                {
                    file_path << "projects/GaryM/test/private_data/crumb_study/hierarchical_mcmc/"
                              << drug_name << "_" << channels[channel_idx] << "_hill_pic50_samples.txt";
                }
                else
                {
                    file_path << "projects/GaryM/test/private_data/crumb_study/standard_mcmc/num_params_" << num_params << "/"
                              << drug_name << "_" << channels[channel_idx] <<"_num_params_" << num_params << "_hill_pic50.txt";
                }
                FileFinder drug_file_name(file_path.str(), RelativeTo::ChasteSourceRoot);
                CrumbDataReader data_reader(drug_file_name, num_params);
                data_readers.push_back(data_reader);
            }

            // Iterate over the dose-response samples of their probability distributions
            for (unsigned sample = 0; sample <  num_samples; sample++)
            {
                // Add some sample drug pIC50 and Hill coefficient for this sample
                for (unsigned channel_idx = 0; channel_idx < channels.size(); channel_idx++)
                {
                    // hERG is hard-coded as Channel index 0 and CaL as channel index 3.
                    if (only_herg_cav && channel_idx!=0u && channel_idx!=3u)
                    {
                        continue;
                    }
                    if (p_model->HasParameter(ap_predict_channels[channel_idx]))
                    {
                        double ic50 = AbstractDataStructure::ConvertPic50ToIc50(data_readers[channel_idx].GetPic50Sample(sample));
                        double conductance_scaling_factor = AbstractDataStructure::CalculateConductanceFactor(concentration,             // Hill
                                                                                                              ic50,                                             // IC50
                                                                                                              data_readers[channel_idx].GetHillSample(sample)); // Hill
                        p_model->SetParameter(ap_predict_channels[channel_idx],default_conductances[channel_idx]*conductance_scaling_factor);

                        std::cout << drug_name << "\tConc = " << concentration << " uM\t" << channels[channel_idx]
                                  << "\tpIC50 = " <<  data_readers[channel_idx].GetPic50Sample(sample)
                                  << "\tHill = " << data_readers[channel_idx].GetHillSample(sample)
                                  << "\tscaling = " << conductance_scaling_factor << std::endl;
                    }
                }

                // Just run a helper method around the usual options (from ApPredict)
                SingleActionPotentialPrediction ap_runner(p_model);
                ap_runner.SuppressOutput();
                ap_runner.SetMaxNumPaces(1000u);
                ap_runner.RunSteadyPacingExperiment();

                if (!ap_runner.DidErrorOccur())
                {
                    *p_file << sample << "\t" << concentration << "\t" << ap_runner.GetApd90() << "\t" << ap_runner.GetApd50() << "\t" << ap_runner.GetCaMax() << "\t" << ap_runner.GetCaMin() << std::endl;
                }
                else
                {
                    *p_file << sample << "\t" << concentration << "\t" << ap_runner.GetErrorMessage() << std::endl;
                    p_model->SetStateVariables(steady_state_variables);
                }
            }
            p_file->close();
            DeleteVector(steady_state_variables);
        }

        PetscTools::IsolateProcesses(false);
        std::cout << "Run complete." << std::endl;
    }
};

#endif // TESTCRUMBPREDICTIONS_HPP_
