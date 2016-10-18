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

#ifndef CRUMBDRUGLIST_HPP_
#define CRUMBDRUGLIST_HPP_

#include "AbstractDataStructure.hpp"

/**
 * Possible CiPA clinical risk categories
 */
typedef enum CipaRiskCategory_
{
    HIGH,
    INTERMEDIATE,
    LOW
} CipaRiskCategory;

/**
 * Helper class to read in data on the drug properties (not screening results, see CrumbDataReader for that).
 */
class CrumbDrugList: public AbstractDataStructure
{
private:
    std::vector<std::string> mDrugNames;
    std::vector<double> mClinicalDose;
    std::vector<CipaRiskCategory> mRiskCategories;

protected:

    /**
     * Read a header line if present.
     *
     * We just say we have read it and then move on.
     */
    bool LoadHeaderLine(std::stringstream& rLine)
    {
        return true;
    }

    /**
     * Method to read the entries of a single line of the data file into this class' member variables.
     * @param rLine  Line to read
     */
    virtual void LoadALine(std::stringstream& rLine)
    {
        std::string name;
        double dose;
        unsigned cipa_cat;

        rLine >> name;
        rLine >> dose; // Data file in nM
        rLine >> cipa_cat;

        mDrugNames.push_back(name);
        mClinicalDose.push_back(dose/1000.0); // Put the dose straight into uM so we don't get confused
        mRiskCategories.push_back(CipaRiskCategory(cipa_cat-1u));
    }

  public:

    /**
     * Constructor
     * @param fileName  Name of the data file to read.
     */
    CrumbDrugList(std::string fileName)
      : AbstractDataStructure()
    {
        LoadDataFromFile(fileName);
    };

    /**
     * Constructor
     * @param rFileFinder  File finder pointing to the data file to read.
     */
    CrumbDrugList(FileFinder& rFileFinder)
      : AbstractDataStructure()
    {
        LoadDataFromFile(rFileFinder.GetAbsolutePath());
    };

    virtual ~CrumbDrugList(){};

    /**
     * @return The number of drugs in the data file.
     */
    unsigned GetNumDrugs(void)
    {
        return mDrugNames.size();
    }

    /**
     * @param drugIndex  The index of the drug (row of the data file on which it appears)
     * @return  The name of the drug
     */
    std::string GetDrugName(unsigned drugIndex)
    {
        assert(drugIndex < GetNumDrugs());
        return mDrugNames[drugIndex];
    }

    /**
     * @param rName  The name of the drug
     * @return  The index in the current drug list.
     */
    unsigned GetDrugIndex(const std::string& rName)
    {
        unsigned idx = UINT_MAX;
        for (unsigned i=0; i<mDrugNames.size(); ++i)
        {
            if (mDrugNames[i] == rName)
            {
                idx = i;
                break;
            }
        }
        if (idx==UINT_MAX)
        {
            EXCEPTION("Drug " << rName << " not found.");
        }
        return idx;
    }

    /**
     * @param rName  Name of the compound
     * @return  Clinical Cmax in microMolar - N.B. Not nM as in the data file.
     */
    double GetClinicalDose(const std::string& rName)
    {
        return mClinicalDose[GetDrugIndex(rName)];
    }

    /**
     * @param rName The name of the compounds
     * @return An enum specifying the risk category as LOW, INTERMEDIATE or HIGH.
     */
    CipaRiskCategory GetRiskCategory(const std::string& rName)
    {
        return mRiskCategories[GetDrugIndex(rName)];
    }

};

#endif // CRUMBDRUGLIST_HPP_
