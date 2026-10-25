#ifndef RTXDI_DI_CGNS_NEIGHBOR_SELECTION_HLSLI
#define RTXDI_DI_CGNS_NEIGHBOR_SELECTION_HLSLI

#include "Rtxdi/DI/CGNSUtils.hlsli"

#ifndef RTXDI_DI_CGNS_MAX_NEIGHBORS
#define RTXDI_DI_CGNS_MAX_NEIGHBORS 32
#endif

/**
 * Weighted Reservoir Sampling structs for CGNS neighbor selection.
 *
 * AChaoReservoir: A-Chao [Chao 1982, see Efraimidis 2015] - running weight sum, probabilistic accept. M=1 only.
 *
 * AESReservoir: A-ES [Efraimidis & Spirakis 2006, see Efraimidis 2015] - key = log(u)/w per candidate, maintain
 *  top-M by evicting the minimum-key slot. Correct for any M.
 *  Implemented with a fixed array of 5 slots (M = 1..5). Increase allocation for M>5.
 *  Decrease allocation to improve performance for a specific count 2 <= M <= 5.
 */

struct RTXDI_DICGNSNeighborSelection
{
    int2 pixels[RTXDI_DI_CGNS_MAX_NEIGHBORS];
    uint count;
};

// ---------------------------------------------------------------------------
// A-Chao: probabilistic accept on running weight sum.
// ---------------------------------------------------------------------------
struct RTXDI_AChaoReservoir
{
    int2  selected;
    float weightSum;

    void init(int2 initialItem)
    {
        selected  = initialItem;
        weightSum = 0.0f;
    }

    // Returns true if candidate was selected.
    bool tryUpdate(int2 candidate, float score, float u)
    {
        weightSum += score;
        if (weightSum > 0.0f && u < score / weightSum)
        {
            selected = candidate;
            return true;
        }
        return false;
    }

    int2 get() { return selected; }
};

// ---------------------------------------------------------------------------
// A-ES (Efraimidis-Spirakis): key = log(u)/w, maintain top-M.
// Fixed capacity of 32 slots (M <= 32)
// only activated when M > 1
// ---------------------------------------------------------------------------
struct RTXDI_AESReservoir
{
    static const uint kMaxCapacity = RTXDI_DI_CGNS_MAX_NEIGHBORS;

    int2  pixels[kMaxCapacity];
    float keys[kMaxCapacity];

    void init(int2 initialItem, uint M)
    {
        for (uint i = 0; i < M; i++)
        {
            pixels[i] = initialItem;
            keys[i]   = -1e38f;
        }
    }

    // Returns true if candidate was inserted (evicted a previous slot).
    bool tryUpdate(int2 candidate, float score, float u, uint M)
    {
        float key    = log(max(u, EPSILON)) / max(score, EPSILON);
        uint  minIdx = 0;
        float minKey = keys[0];

        // Linear probe; can be optimized with a min heap for larger M.
        for (uint i = 1; i < M; i++)
        {
            if (keys[i] < minKey) { minKey = keys[i]; minIdx = i; }
        }
        if (key > minKey)
        {
            pixels[minIdx] = candidate;
            keys[minIdx]   = key;
            return true;
        }
        return false;
    }

    int2 get(uint i) { return pixels[i]; }
};

float RTXDI_CGNSComputeCandidateScore(
    uint2 pixelPosition,
    uint sampleIndex,
    RTXDI_RuntimeParameters rParams,
    RTXDI_DISpatialResamplingParameters sparams,
    bool centerValid,
    float3 centerPos,
    float3 centerNormal,
    float scale,
    out int2 candidatePixel)
{
    uint sampleIdx = sampleIndex & rParams.neighborOffsetMask;
    candidatePixel = int2(round(float2(pixelPosition) + float2(RTXDI_NEIGHBOR_OFFSETS_BUFFER[sampleIdx].xy) * sparams.samplingRadius));

    int2 clampedPixel = RAB_ClampSamplePositionIntoView(candidatePixel, false);
    if (candidatePixel.x != clampedPixel.x || candidatePixel.y != clampedPixel.y)
        return 0.0f;

    if (candidatePixel.x == int(pixelPosition.x) && candidatePixel.y == int(pixelPosition.y))
        return 0.0f;

    RAB_Surface candidateSurface = RAB_GetGBufferSurface(candidatePixel, false);
    bool candidateValid = RAB_IsSurfaceValid(candidateSurface);
    float3 candidatePos = candidateValid ? RAB_GetSurfaceWorldPos(candidateSurface) : float3(0.0f, 0.0f, 0.0f);
    float3 candidateNormal = candidateValid ? RAB_GetSurfaceNormal(candidateSurface) : float3(0.0f, 0.0f, 0.0f);

    return RTXDI_CGNSComputeNeighborWeight(
        centerValid, candidateValid,
        centerPos, centerNormal,
        candidatePos, candidateNormal,
        scale, max(sparams.cgnsNormalBeta, 0.0f));
}

void RTXDI_DISelectCGNSNeighbors(
    uint2 pixelPosition,
    RAB_Surface centerSurface,
    inout RTXDI_RandomSamplerState rng,
    RTXDI_RuntimeParameters rParams,
    RTXDI_DISpatialResamplingParameters sparams,
    uint numSpatialSamples,
    out RTXDI_DICGNSNeighborSelection selection)
{
    selection.count = 0;

    numSpatialSamples = min(numSpatialSamples, RTXDI_DI_CGNS_MAX_NEIGHBORS);
    if (numSpatialSamples == 0)
        return;

    bool centerValid = RAB_IsSurfaceValid(centerSurface);
    float3 centerPos = centerValid ? RAB_GetSurfaceWorldPos(centerSurface) : float3(0.0f, 0.0f, 0.0f);
    float3 centerNormal = centerValid ? RAB_GetSurfaceNormal(centerSurface) : float3(0.0f, 0.0f, 0.0f);
    float centerDist = centerValid ? RAB_GetSurfaceLinearDepth(centerSurface) : 0.0f;
    float scale = RTXDI_CGNSComputeNeighborScale(max(sparams.cgnsScaleSolidAngle, EPSILON), centerDist);

    uint candidateCount = clamp(sparams.cgnsNumCandidates, 1u, 128u);
    uint startIndex = uint(RTXDI_GetNextRandom(rng) * rParams.neighborOffsetMask);

    if (numSpatialSamples == 1)
    {
        // A-Chao WRS algorithm is simplest for M=1
        RTXDI_AChaoReservoir res;
        res.init(int2(-1, -1));

        for (uint k = 0; k < candidateCount; k++)
        {
            int2 candidatePixel;
            float score = RTXDI_CGNSComputeCandidateScore(pixelPosition, startIndex + k, rParams, sparams, centerValid, centerPos, centerNormal, scale, candidatePixel);
            if (score <= 0.0f)
                continue;

            res.tryUpdate(candidatePixel, score, RTXDI_GetNextRandom(rng));

            if (sparams.enableCGNSEarlyStopping && score >= sparams.cgnsEarlyStopCutoff)
                break;
        }

        int2 selectedPixel = res.get();
        if (selectedPixel.x >= 0)
        {
            selection.pixels[0] = selectedPixel;
            selection.count = 1;
        }
    }
    else
    {
        // A-ES: key = log(u)/w per candidate, maintain top-M. Unbiased for any M.
        RTXDI_AESReservoir res;
        res.init(int2(-1, -1), numSpatialSamples);

        uint highQualityCount = 0;
        for (uint k = 0; k < candidateCount; k++)
        {
            int2 candidatePixel;
            float score = RTXDI_CGNSComputeCandidateScore(
                pixelPosition, startIndex + k, rParams, sparams, centerValid, centerPos, centerNormal, scale, candidatePixel);
            if (score <= 0.0f)
                continue;

            res.tryUpdate(candidatePixel, score, RTXDI_GetNextRandom(rng), numSpatialSamples);

            if (sparams.enableCGNSEarlyStopping && score >= sparams.cgnsEarlyStopCutoff)
            {
                if (++highQualityCount >= numSpatialSamples)
                    break;
            }
        }

        for (uint j = 0; j < numSpatialSamples; j++)
        {
            int2 selectedPixel = res.get(j);
            if (selectedPixel.x >= 0)
            {
                selection.pixels[selection.count] = selectedPixel;
                selection.count += 1;
            }
        }
    }
}

#endif // RTXDI_DI_CGNS_NEIGHBOR_SELECTION_HLSLI
