#ifndef RTXDI_DI_PVALUE_NEIGHBOR_SELECTION_HLSLI
#define RTXDI_DI_PVALUE_NEIGHBOR_SELECTION_HLSLI

#include "Rtxdi/DI/CGNSNeighborSelection.hlsli"

float RTXDI_PValueComputeCandidateScore(
    uint2 pixelPosition,
    uint sampleIndex,
    RAB_Surface centerSurface,
    RTXDI_DIReservoir centerSample,
    RTXDI_RuntimeParameters rParams,
    RTXDI_ReservoirBufferParameters reservoirParams,
    uint sourceBufferIndex,
    RTXDI_DISpatialResamplingParameters sparams,
    out int2 candidatePixel)
{
    uint sampleIdx = sampleIndex & rParams.neighborOffsetMask;
    candidatePixel = int2(round(float2(pixelPosition) + float2(RTXDI_NEIGHBOR_OFFSETS_BUFFER[sampleIdx].xy) * sparams.samplingRadius));

    int2 clampedPixel = RAB_ClampSamplePositionIntoView(candidatePixel, false);
    if (candidatePixel.x != clampedPixel.x || candidatePixel.y != clampedPixel.y)
        return 0.0f;

    if (candidatePixel.x == int(pixelPosition.x) && candidatePixel.y == int(pixelPosition.y))
        return 0.0f;

    RTXDI_ActivateCheckerboardPixel(candidatePixel, false, rParams.activeCheckerboardField);

    int2 clampedActivatedPixel = RAB_ClampSamplePositionIntoView(candidatePixel, false);
    if (candidatePixel.x != clampedActivatedPixel.x || candidatePixel.y != clampedActivatedPixel.y)
        return 0.0f;

    if (candidatePixel.x == int(pixelPosition.x) && candidatePixel.y == int(pixelPosition.y))
        return 0.0f;

    RAB_Surface candidateSurface = RAB_GetGBufferSurface(candidatePixel, false);
    if (!RAB_IsSurfaceValid(candidateSurface))
        return 0.0f;

    if (!RTXDI_IsValidNeighbor(
        RAB_GetSurfaceNormal(centerSurface), RAB_GetSurfaceNormal(candidateSurface),
        RAB_GetSurfaceLinearDepth(centerSurface), RAB_GetSurfaceLinearDepth(candidateSurface),
        sparams.normalThreshold, sparams.depthThreshold))
        return 0.0f;

    uint2 candidateReservoirPos = RTXDI_PixelPosToReservoirPos(uint2(candidatePixel), rParams.activeCheckerboardField);

    float4 candidateStats = RTXDI_LoadDIReservoirStats(reservoirParams, candidateReservoirPos, sourceBufferIndex);

    if (candidateStats.z <= 0.0 || candidateStats.w <= 0.0)
        return 0.0f;

    if (sparams.discountNaiveSamples && candidateStats.w <= RTXDI_NAIVE_SAMPLING_M_THRESHOLD)
        return 0.0f;

    if (centerSample.statM <= 0.0)
        return 0.0f;

    float pValue = RTXDI_CalculatePValue(centerSample, candidateStats);
    float threshold = clamp(sparams.pValueThreshold, 0.0f, 1.0f);

    return (pValue >= threshold) ? pValue : 0.0f;
}

void RTXDI_DISelectPValueNeighbors(
    uint2 pixelPosition,
    RAB_Surface centerSurface,
    RTXDI_DIReservoir centerSample,
    inout RTXDI_RandomSamplerState rng,
    RTXDI_RuntimeParameters rParams,
    RTXDI_ReservoirBufferParameters reservoirParams,
    uint sourceBufferIndex,
    RTXDI_DISpatialResamplingParameters sparams,
    uint numSpatialSamples,
    out RTXDI_DICGNSNeighborSelection selection)
{
    selection.count = 0;

    numSpatialSamples = min(numSpatialSamples, RTXDI_DI_CGNS_MAX_NEIGHBORS);
    if (numSpatialSamples == 0)
        return;

    uint candidateCount = clamp(sparams.pValueNumCandidates, 1u, 128u);
    float earlyStopThreshold = clamp(sparams.pValueEarlyStopThreshold, 0.0f, 1.0f);
    uint startIndex = uint(RTXDI_GetNextRandom(rng) * rParams.neighborOffsetMask);

    if (numSpatialSamples == 1)
    {
        RTXDI_AChaoReservoir res;
        res.init(int2(-1, -1));

        for (uint k = 0; k < candidateCount; k++)
        {
            int2 candidatePixel;
            float score = RTXDI_PValueComputeCandidateScore(
                pixelPosition, startIndex + k, centerSurface, centerSample, rParams,
                reservoirParams, sourceBufferIndex, sparams, candidatePixel);
            if (score <= 0.0f)
                continue;

            res.tryUpdate(candidatePixel, score, RTXDI_GetNextRandom(rng));

            if (sparams.enableCGNSEarlyStopping && score >= earlyStopThreshold)
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
        RTXDI_AESReservoir res;
        res.init(int2(-1, -1), numSpatialSamples);

        uint highQualityCount = 0;
        for (uint k = 0; k < candidateCount; k++)
        {
            int2 candidatePixel;
            float score = RTXDI_PValueComputeCandidateScore(
                pixelPosition, startIndex + k, centerSurface, centerSample, rParams,
                reservoirParams, sourceBufferIndex, sparams, candidatePixel);
            if (score <= 0.0f)
                continue;

            res.tryUpdate(candidatePixel, score, RTXDI_GetNextRandom(rng), numSpatialSamples);

            if (sparams.enableCGNSEarlyStopping && score >= earlyStopThreshold)
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

#endif // RTXDI_DI_PVALUE_NEIGHBOR_SELECTION_HLSLI
