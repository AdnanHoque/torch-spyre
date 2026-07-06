# Focused Deeptools test summary

Pod: adnan-spyre-dev-pf
Torch SHA: 3f13d2f9fd8b14a9efa986cabd9b1de038faf122
Deeptools SHA: a4930be14b6e7d01f7447b7692a79a20487c09c3
Test directory: /home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/runs/deeptools_tests_chunked_4slot_20260706_071233

## Results

- LayoutAllgatherRestickify.* rc: 0
- CoreWorkDivIncomptLxRelayout* rc: 0

## LayoutAllgatherRestickify tail

[       OK ] LayoutAllgatherRestickify.emitsBuf21MatmulOperandBroadcastPlanArtifact (1 ms)
[ RUN      ] LayoutAllgatherRestickify.matmulOperandBroadcastRecordsLayoutConversionContract
[       OK ] LayoutAllgatherRestickify.matmulOperandBroadcastRecordsLayoutConversionContract (1 ms)
[ RUN      ] LayoutAllgatherRestickify.matmulOperandBroadcastAcceptsGatherThenRestickifyRealization
[       OK ] LayoutAllgatherRestickify.matmulOperandBroadcastAcceptsGatherThenRestickifyRealization (0 ms)
[ RUN      ] LayoutAllgatherRestickify.matmulOperandBroadcastRejectsResidentReplication
[       OK ] LayoutAllgatherRestickify.matmulOperandBroadcastRejectsResidentReplication (0 ms)
[ RUN      ] LayoutAllgatherRestickify.movementPlanRejectsNonFlashProducer
[       OK ] LayoutAllgatherRestickify.movementPlanRejectsNonFlashProducer (0 ms)
[ RUN      ] LayoutAllgatherRestickify.movementPlanRejectsIncompleteRename
[       OK ] LayoutAllgatherRestickify.movementPlanRejectsIncompleteRename (0 ms)
[ RUN      ] LayoutAllgatherRestickify.mismatchedGroupsUseCoordinateRepartition
[       OK ] LayoutAllgatherRestickify.mismatchedGroupsUseCoordinateRepartition (0 ms)
[ RUN      ] LayoutAllgatherRestickify.synthesizesRenamedOneDToTwoDRepartitionPlan
[       OK ] LayoutAllgatherRestickify.synthesizesRenamedOneDToTwoDRepartitionPlan (0 ms)
[----------] 27 tests from LayoutAllgatherRestickify (4 ms total)

[----------] Global test environment tear-down
[==========] 27 tests from 1 test suite ran. (4 ms total)
[  PASSED  ] 27 tests.

## CoreWorkDivIncomptLxRelayout tail

Note: Google Test filter = DxpTestFixture.CoreWorkDivIncomptLxRelayout*
[==========] Running 2 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 2 tests from DxpTestFixture
[ RUN      ] DxpTestFixture.CoreWorkDivIncomptLxRelayout
[       OK ] DxpTestFixture.CoreWorkDivIncomptLxRelayout (82 ms)
[ RUN      ] DxpTestFixture.CoreWorkDivIncomptLxRelayoutCardinality
[       OK ] DxpTestFixture.CoreWorkDivIncomptLxRelayoutCardinality (106 ms)
[----------] 2 tests from DxpTestFixture (189 ms total)

[----------] Global test environment tear-down
[==========] 2 tests from 1 test suite ran. (189 ms total)
[  PASSED  ] 2 tests.
