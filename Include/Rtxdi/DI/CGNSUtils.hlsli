#ifndef RTXDI_DI_CGNS_UTILS_HLSLI
#define RTXDI_DI_CGNS_UTILS_HLSLI

/**
 * Compatibility scoring for spatial neighbor selection.
 * Used by SelectNeighbors.cs.slang to score candidate pixels before WRS selection.
 *
 * Heuristic: h = h_position * h_normal  (paper §4.3)
 *   h_position = exp(-||x - y|| / s),      s = sqrt(Ω/π) * queryDist
 *   h_normal   = max(n_x · n_y, 0)^β
 *
 * Paper defaults: Ω = 0.05 sr, β = 8.
 */

#ifndef EPSILON
#define EPSILON 1e-8f
#endif

static const float RTXDI_CGNS_PI = 3.14159265f;

// s = sqrt(Ω/π) * d  (characteristic world-space radius at distance d, paper §4.3)
// Controls position falloff radius ensuring the heuristic is agnostic to scene scale.
float RTXDI_CGNSComputeNeighborScale(float solidAngle, float dist)
{
    return max(EPSILON, sqrt(solidAngle / RTXDI_CGNS_PI) * dist);
}

float RTXDI_CGNSComputeNeighborWeight(
    bool   queryValid,
    bool   candidateValid,
    float3 queryPos,
    float3 queryNormal,
    float3 candidatePos,
    float3 candidateNormal,
    float  scale,      // s = sqrt(Ω/π) * queryDist
    float  normalBeta  // β: exponent on the normal compatibility term
)
{
    // Special handling for invalid hits (e.g. environment maps)
    if (!queryValid && !candidateValid)
        return 1.0f;   // both env map hits — encourage reuse

    if (!queryValid || !candidateValid)
        return EPSILON; // one surface, one env map — discourage

    float positionScore = exp(-distance(queryPos, candidatePos) / scale);
    float normalScore   = max(0.0f, dot(queryNormal, candidateNormal));

    return clamp(positionScore * pow(normalScore, normalBeta), EPSILON, 1.0f);
}

#endif // RTXDI_DI_CGNS_UTILS_HLSLI
