import sys, re
from pathlib import Path, PurePath
from dataclasses import dataclass, field
# Find path to root folder
rootFolder = 'sounds-phd'
pathToRoot = Path(__file__)
while PurePath(pathToRoot).name != rootFolder:
    pathToRoot = pathToRoot.parent
sys.path.append(f'{pathToRoot}/01_algorithms/01_NR/02_distributed')
from danse_testing import *
from danse_utilities.classes import SROsTimeVariation

@dataclass
class TestsParams:
    pathToASC: str = '' # path to acoustic scenario to consider
    sigDur: float = 15. # signal duration [s]
    computeCentralised: bool = False    # if True, compute centralised
    baseSNR: float = 0.  # SNR at ref. mic. of node 1
    deltaSROs: list[int] = field(default_factory=list)  # list of SROs
        # ^^^ 2 simulations per element: w/ and w/out comp.
    exportBasePath: str = ''        # path to exported files
    timeVaryingSROs: bool = False   # if True, consider time-varying SROs
    noiseType: str = 'white'        # type of noise generated by noise source(s)
    DFTsize: int = 1024             # DFT size (number of bins)
    computeSDR: bool = False   
        # ^^^ if True, compute the noise-free DANSE filtering outcome to
        # compute the SI-SDR.


ASCBASEPATH = '02_data/01_acoustic_scenarios/for_submissions/icassp2023'
# EXPORTPATH = '01_algorithms/01_NR/02_distributed/res/for_submissions/icassp2023/'
EXPORTPATH = '01_algorithms/01_NR/02_distributed/res/for_submissions/icassp2023/test_postReviews/'
PARAMS = TestsParams(
    # vvv 20 cm mic. spacing vvv
    pathToASC=f'{ASCBASEPATH}/J4Mk[1_3_2_5]_Ns1_Nn2/AS18_RT150ms',
    # vvv 10 cm mic. spacing vvv
    # pathToASC=f'{ASCBASEPATH}/J4Mk[1_3_2_5]_Ns1_Nn2/AS37_RT150ms',
    sigDur=15,
    # sigDur=5,
    baseSNR=-3,  # <-- = -3 (dB) in ICASSP2023 submission
    computeCentralised=True,
    deltaSROs=[20,50,200],
    # deltaSROs=[200],
    # deltaSROs=[20],
    # deltaSROs=[],  # uncomment this to only compute the centralised case
    exportBasePath=EXPORTPATH,
    timeVaryingSROs=SROsTimeVariation(
        timeVarying=False,
        varRange=10,
        probVarPerSample=0.0001,
        transition='step'
    ),
    noiseType='white',      # white noise (localised)
    # noiseType='white_diffuse',      # white noise (diffuse)
    # noiseType='ssn',        # speech-shaped noise (localised)
    # noiseType='ssn_diffuse',        # speech-shaped noise (diffuse)
    # noiseType='babble',     # babble noise (localised)
    # noiseType='babble_diffuse',     # babble noise (diffuse)
    # noiseType='nonstationaryspeech',      # non-stationary speech (localised)
    # DFTsize=512,    # DFT size (number of bins)
    DFTsize=2048,    # DFT size (number of bins)
    #
    computeSDR = True   # compute SI-SDR
)


def main():

    # Derive the number of simulations to conduct
    nTest = 2 * len(PARAMS.deltaSROs)
    if PARAMS.computeCentralised:
        nTest += 1

    # Derive the number of nodes
    nNodes = int(Path(PARAMS.pathToASC).parent.stem[1])

    # Adjust parameters depending on test
    for ii in range(nTest):
        # Compute SRO-free centralised estimate last, if asked
        if PARAMS.computeCentralised and ii == nTest - 1:
            params = dict([  # centralised case
                ('listedSROs', [0] * nNodes),
                ('comp', False),
                ('bc', 'wholeChunk'),
                ('computeCentrEstimate', True)
            ])
        else:
            lsros = [
                ((-1)**jj) * ((jj + 1) // 2) * PARAMS.deltaSROs[ii // 2]\
                    for jj in range(nNodes)
            ]
            if ii % 2 == 0:
                params = dict([  # do compensate SROs
                    ('listedSROs', lsros),
                    ('comp', True),
                    ('bc', 'samplePerSample'),
                    ('computeCentrEstimate', False)
                ])
            else:
                params = dict([  # don't compensate SROs
                    ('listedSROs', lsros),
                    ('comp', False),
                    ('bc', 'wholeChunk'),
                    ('computeCentrEstimate', False)
                ])
        
        # Commmon parameters in all cases
        params['ASC'] = PARAMS.pathToASC
        params['timeVaryingSROs'] = PARAMS.timeVaryingSROs
        params['noiseType'] = PARAMS.noiseType
        params['sigDur'] = PARAMS.sigDur
        params['baseSNR'] = PARAMS.baseSNR
        params['DFTsize'] = PARAMS.DFTsize
        params['computeNoiseFreeForSDR'] = PARAMS.computeSDR

        run_simul(params, PARAMS.exportBasePath)


def run_simul(params, exportBasePath):

    # Infer number of noise sources from ASC path name
    # (https://stackoverflow.com/a/28526377)
    nNoiseS = int(re.findall(
        "\d+", Path(params['ASC']).parent.stem[-3:]
    )[0])
    # Determine which noise to use
    if 'white' in params['noiseType']:
        noiseFiles = ['whitenoise_signal_1.wav', 'whitenoise_signal_2.wav']
    elif 'ssn' in params['noiseType']:
        noiseFiles = ['ssn/ssn_speech1.wav', 'ssn/ssn_speech2.wav']
    elif 'babble' in params['noiseType']:
        noiseFiles = [f'babble/babble{ii + 1}.wav' for ii in range(nNoiseS)]
    elif 'nonstationaryspeech' in params['noiseType']:
        p = Path(f'{pathToRoot}/02_data/00_raw_signals/02_noise/speech')\
            .glob('**/*')  # https://stackoverflow.com/a/40216619
        files = [x for x in p if x.is_file()]
        noiseFiles = [f'speech/{files[ii].stem}.wav' for ii in range(nNoiseS)]
    # Make diffuse or not
    diffuseNoise = 'diffuse' in params['noiseType']

    # Generate test parameters
    danseTestingParams = sro_testing.DanseTestingParameters(
        # General hyperparameters
        writeOver=True,    # if True, the script will overwrite existing data
        #
        ascBasePath=f'{pathToRoot}/02_data/01_acoustic_scenarios/validations',
        signalsPath=f'{Path(__file__).parent}/validations/signals',
        #
        # vvvvvvvvvvvvvv
        specificAcousticScenario=[f'{pathToRoot}/{params["ASC"]}'],
        #
        fs=16000,
        specificDesiredSignalFiles=[
            f'{pathToRoot}/02_data/00_raw_signals/01_speech/{file}'\
                for file in ['speech1.wav', 'speech2.wav']
        ],
        specificNoiseSignalFiles=[
            f'{pathToRoot}/02_data/00_raw_signals/02_noise/{file}'\
                for file in noiseFiles
        ],
        # vvvvvvvvvvvvvv
        diffuseNoise=diffuseNoise,
        # vvvvvvvvvvvvvv
        sigDur=params['sigDur'],
        # vvvvvvvvvvvvvv
        baseSNR=params['baseSNR'],
        #
        SROsParams=TestSROs(
            type='specific',
            # vvvvvvvvvvvvvv
            listedSROs=params['listedSROs'],
        ),
        #
        timeBtwExternalFiltUpdates=3.,
        #
        # vvvvvvvvvvvvvv
        broadcastScheme=params['bc'],
        performGEVD=1,
        #
        computeLocalEstimate=True,
        # vvvvvvvvvvvvvv
        computeCentrEstimate=params['computeCentrEstimate'],
        # computeCentrEstimate=True,    # BUG: DEBUGGING 20230119
        computeNoiseFreeForSDR=params['computeNoiseFreeForSDR'],
        #
        asynchronicity=SamplingRateOffsets(
            plotResult=True,
            # vvvvvvvvvvvvvv
            compensateSROs=params['comp'],
            estimateSROs='CohDrift',
            cohDriftMethod=CohDriftSROEstimationParameters(
                loop='open',
                # startAfterNupdates=30  # BUG: DEBUGGING 20230118
            ),
            # vvvvvvvvvvvvvv
            timeVaryingSROs=params['timeVaryingSROs']
        ),
        printouts=PrintoutsParameters(
            progressPrintingInterval=0.5
        ),
        #
        DFTsize=params['DFTsize']
    )

    # Run test
    sro_testing.go(danseTestingParams, exportBasePath)


if __name__=='__main__':
    sys(exit(main()))