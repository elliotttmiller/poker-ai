# PokerMind Performance Analysis Report

Generated: 2025-09-06 13:54:29

## Executive Summary

The CognitiveCore.make_decision method was profiled across 300 decision scenarios to identify performance characteristics and optimization opportunities.

### Key Metrics
- **Average Decision Time**: 6.320ms
- **Median Decision Time**: 6.318ms  
- **Min/Max Decision Time**: 6.307ms / 6.465ms
- **Standard Deviation**: 0.018ms
- **Target Performance**: <10ms per decision (✅ MET)

## Scenario Analysis

### PREFLOP Scenario ✅
- Average: 6.324ms
- Median: 6.318ms
- Range: 6.307ms - 6.465ms
- Std Dev: 0.029ms
- Iterations: 100

### FLOP Scenario ✅
- Average: 6.318ms
- Median: 6.318ms
- Range: 6.307ms - 6.360ms
- Std Dev: 0.006ms
- Iterations: 100

### RIVER Scenario ✅
- Average: 6.320ms
- Median: 6.319ms
- Range: 6.314ms - 6.368ms
- Std Dev: 0.006ms
- Iterations: 100

## Performance Bottlenecks

## Optimization Recommendations

✅ Performance target met: 6.32ms average

💡 Consider implementing result caching for expensive operations

💡 Optimize ONNX model inference sessions

💡 Use numpy operations instead of Python loops where possible

💡 Profile memory allocations to reduce GC pressure

## Technical Implementation Notes

### Current Architecture
The CognitiveCore implements a dual-process architecture:
- **System 1 (Parallel)**: GTO Core, Hand Strength Estimator, Heuristics Engine, Opponent Modeler
- **System 2 (Sequential)**: Confidence-weighted synthesis and meta-cognitive adjustments

### Performance Characteristics
- **Parallel Processing**: System 1 modules run concurrently with 0.5s timeout
- **Confidence Scoring**: Enhanced Phase 5 confidence calculations add minimal overhead
- **Memory Usage**: Efficient with deque-based action histories and numpy operations

### Optimization Opportunities
1. **Model Inference**: ONNX runtime optimization for GTO Core and Hand Strength models
2. **Caching**: Implement result caching for repeated game states
3. **Memory Management**: Optimize object creation in hot paths
4. **Threading**: Optimize ThreadPoolExecutor configuration

## Raw Performance Data

```json
{
  "preflop": {
    "avg_time_ms": 6.323546579999402,
    "iterations": 100,
    "target_met": true
  },
  "flop": {
    "avg_time_ms": 6.317596000002368,
    "iterations": 100,
    "target_met": true
  },
  "river": {
    "avg_time_ms": 6.320316840008218,
    "iterations": 100,
    "target_met": true
  }
}
```
