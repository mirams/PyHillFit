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

#ifndef CRUMBDATAREADER_HPP_
#define CRUMBDATAREADER_HPP_

#include <fstream>
#include <vector>

#include "UblasVectorInclude.hpp"
#include "FileFinder.hpp"
#include "AbstractDataStructure.hpp"

/**
 * A class which is designed to read in data on drugs in teh Crumb
 * study, for which we have already analysed dose-response data
 * and inferred possible IC50 and Hill distribution parameters.
 */
class CrumbDataReader: public AbstractDataStructure
{
protected:
    /**
     * A method which hard-codes the format of this file
     * @param rLine  a line in stringsteam format.
     */
    void LoadALine(std::stringstream& rLine)
    {
        if (mNumParams==2u)
        {
            double hill;
            rLine >> hill;
            mHillSamples.push_back(hill);
        }

        double pIc50;
        rLine >> pIc50;
        mPic50Samples.push_back(pIc50);
    }

    bool LoadHeaderLine(std::stringstream& rLine)
    {
        // There is a header line...
        return true;
    }

    /** A pair of parameters for logistic pIC50 distributions,
     * indexed over drug first, and then channel (7) */
    std::vector<double> mPic50Samples;

    /** A pair of parameters for log-logistic Hill distributions,
     * indexed over drug first, and then channel (7) */
    std::vector<double> mHillSamples;

    /**
     * The number of parameters that are in the samples
     * (1 = pIC50, 2 = pIC50 and Hill)
     */
    unsigned mNumParams;

    /**
     * Default Constructor (empty)
     */
    CrumbDataReader(){};

public:

    /**
     * Constructor. Data will be immediately loaded into memory by the constructor.
     *
     * @param fileName  the name of a file to load (relative or absolute).
     * @param numParams  The number of parameters that are in the samples (1 = pIC50, 2 = pIC50 and Hill)
     */
    CrumbDataReader(std::string fileName, unsigned numParams)
      : AbstractDataStructure(),
        mNumParams(numParams)
    {
        assert(mNumParams==1u || mNumParams==2u);
        LoadDataFromFile(fileName, 1u);
    };

    /**
     * Constructor. Data will be immediately loaded into memory by the constructor.
     *
     * @param rFileFinder  a file finder pointing to the data file to load.
     * @param numParams  The number of parameters that are in the samples (1 = pIC50, 2 = pIC50 and Hill)
     */
    CrumbDataReader(FileFinder& rFileFinder, unsigned numParams)
      : AbstractDataStructure(),
        mNumParams(numParams)
    {
        assert(mNumParams==1u || mNumParams==2u);
        LoadDataFromFile(rFileFinder.GetAbsolutePath(), 1u);
    };

    /**
     * Destructor (empty)
     */
    virtual ~CrumbDataReader(){};

    /**
     * Return the IC50 value samples associated with a particular channel.
     *
     * @param the index of the sample in this list
     * @return the IC50 value for a certain drug on a certain channel.
     */
    double GetPic50Sample(const unsigned& rIndex)
    {
        if (rIndex >= mPic50Samples.size())
        {
            EXCEPTION("Sample index " << rIndex << " requested, but the number of samples in the data file is " << mPic50Samples.size() << ".");
        }
        return mPic50Samples[rIndex];
    }

    /**
     * Return the hill coefficient samples associated with this drug and this channel's dose-reponse curve
     *
     * @param the index of the sample in this list
     * @return the hill coefficient
     */
    double GetHillSample(const unsigned& rIndex)
    {
        if (mNumParams==1u)
        {
            return 1.0;
        }

        if (rIndex >= mHillSamples.size())
        {
            EXCEPTION("Sample index " << rIndex << " requested, but the number of samples in the data file is " << mHillSamples.size() << ".");
        }

        return mHillSamples[rIndex];
    }
};
#endif // CRUMBDATAREADER_HPP_
