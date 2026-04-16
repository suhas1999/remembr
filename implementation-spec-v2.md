# ReMEmbR v2: Intelligent Visual Memory for Robot Navigation QA

## The Problem with ReMEmbR

ReMEmbR captions every 3-second video chunk and stores all of them — a 25-minute walk produces ~500 entries of mostly redundant captions like "a hallway with lights" repeated dozens of times. This causes two failures: retrieval dilution (relevant results buried under duplicates competing for top-K slots) and information loss (captions are a lossy bottleneck — the robot sees a water fountain but the caption says "silver machine," and that detail is gone forever). Questions about visual details like colors, sign text, clothing, or movement direction are structurally unanswerable because no images are stored.

## Our Approach

Store fewer, richer memories. Each memory includes the actual keyframe image, a detailed VLM-generated caption, visual and text embeddings, timestamp, and GPS position. At query time, retrieve relevant memories through hybrid search and let a VLM see the actual stored images to answer. An LLM agent with general-purpose tools handles all question types without hardcoded routing.

## Memory Building Pipeline

**Step 1: Batch encode all frames with SigLIP.** Extract all frames from the video, encode them all at once using SigLIP-SO400M. Every frame gets a 1152-dim visual embedding. This is pure computation with no API calls and takes ~30 seconds for 8000 frames on an H100.

**Step 2: Iterate through frames with a two-stage filter.**

The first stage is a fast cosine similarity check. Each frame's SigLIP embedding is compared against the current anchor embedding (the frame where the current scene started). If similarity is above 0.9, the frame is near-identical to the anchor — skip it entirely. This eliminates the vast majority of frames instantly without any VLM calls.

The second stage runs on the frames that pass the first filter. For each, retrieve the top-3 most visually similar stored entries from the vector database using the frame's SigLIP embedding. These retrieved frames, combined with the current anchor frame and the previous anchor frame (from the scene just before the current one), form the comparison set. The current frame plus this comparison set are sent to the VLM. The previous anchor is included to give the VLM temporal context about where the robot just came from — this helps it write captions that capture transitions ("after leaving the outdoor courtyard, the robot entered a lobby") and reduces hallucinated descriptions.

The VLM is asked: does this frame add any information not already captured in the stored images? It responds with whether the frame adds information, whether it's a different location, whether it's a revisit of an earlier location, a detailed caption, and any readable text/signs.

The retrieval step during building is critical — it catches revisits. Without it, when the robot returns to Hallway A after visiting Room B, the system would store a duplicate because the recent context no longer contains the original Hallway A frames. With retrieval, the original Hallway A frame surfaces as visually similar, and the VLM recognizes it as already captured.

**Step 3: Store or skip based on VLM decision.**

If the VLM says the frame adds information: save the keyframe image to disk, compute a BGE caption embedding, and insert the full entry (image path, SigLIP embedding, BGE embedding, caption, timestamp, position, location-change flag) into the vector database. If it's a different location, update the anchor (and record the old anchor as the previous anchor for the next scene's captioning context). If it's the same scene but with new content (a person appeared, a sign became readable), keep the anchor unchanged.

If the VLM says it's a revisit of an earlier stored location: store a lightweight revisit marker that reuses the original entry's image and caption but records the new timestamp and position. This costs almost nothing in storage but preserves temporal presence for duration questions. Update the anchor to the revisited location so subsequent frames compare correctly.

If the VLM says no new information: skip entirely.

**Expected output:** A 25-minute walk produces ~65-80 stored entries (~55-70 unique images on disk, remainder are revisit markers reusing existing images). This is an 85% reduction from ReMEmbR's ~500 entries, with every entry carrying actual visual evidence.

## Storage Schema

Each memory entry in Milvus (reusing ReMEmbR's existing database setup) has two searchable vector fields: a 1152-dim SigLIP visual embedding and a 768-dim BGE caption embedding. The payload stores the caption text, timestamp, GPS position, image path on disk, and a location-change flag. Keyframe images are stored as JPEG at 768px max resolution, ~50-80KB each.

Duration is computed at query time from entry timestamps rather than stored explicitly. When the agent needs to answer "how long were you in the building," it retrieves the sequence of stored entries, identifies the first building entry and the last building entry (or first non-building entry after), and computes the difference. With ~70 entries over 25 minutes the temporal resolution is ~20 seconds, well within the benchmark's 2-minute correctness threshold and without the edge cases that come from tracking scene boundaries implicitly during building.

## Retrieval (Query Time)

Two parallel retrieval channels fused with Reciprocal Rank Fusion. Caption embedding search: encode the query with BGE, search against stored caption embeddings — this is the workhorse handling most object and scene queries. Visual embedding search: encode the query with SigLIP's text encoder, search against stored SigLIP image embeddings — this is the safety net that bypasses captions entirely, catching objects the captioner described poorly ("water fountain" as query matches the image of a water fountain even when the caption says "silver machine").

Results from both channels are merged using RRF and the top-K are returned. For spatial queries containing words like "closest" or "nearest," results are re-ranked by distance to the robot's current position.

**Context expansion:** For every retrieved memory, also fetch its temporally adjacent stored entries (the one stored immediately before and immediately after by timestamp). These neighbor frames give the answering VLM motion and spatial context — seeing what the robot was approaching and what it just left. A single keyframe is a snapshot; a triplet (before → match → after) is a trajectory. This is cheap (just two extra DB lookups per result) and significantly helps trajectory questions ("which direction did you turn"), spatial reasoning ("near the stairs"), and disambiguation ("which hallway — the one before or after the lobby").

## Query Agent

A tool-calling LLM agent with four general-purpose tools. `search_memory` runs the hybrid retrieval. `search_near_position` finds memories near a GPS coordinate. `get_nearby_in_time` returns all stored memories within a time window around a timestamp, ordered chronologically — essential for trajectory, sequence, and duration questions. `examine_keyframes` loads stored images from disk and sends them to the VLM with a question — this is the core capability ReMEmbR lacks, enabling the system to actually see colors, read signs, count objects, and understand movement direction. `answer` provides the final response with optional position/time/duration fields. There is no question-type classification or if-else routing. The LLM reads tool descriptions and decides its own strategy.

## Model Choices

SigLIP-SO400M-patch14-384 for visual encoding (shared between memory building and retrieval). BGE-base-en-v1.5 for caption text encoding. GPT-5.4-mini for the VLM during memory building (judge decisions and captioning). GPT-4o for the agent LLM and query-time visual answering (same as ReMEmbR baseline for clean comparison). Milvus for the vector database (reusing ReMEmbR's existing setup).

## Why This Wins

ReMEmbR stores 500 lossy text summaries and hopes the captioner mentioned the right details. We store 70 actual visual memories with rich captions and multi-modal retrieval. The ~10% of benchmark questions about visual details (colors, sign text, clothing, hand gestures) are structurally unanswerable by ReMEmbR — they have no images to look at. Trajectory questions work because `get_nearby_in_time` retrieves ordered frame sequences that the VLM can inspect for turns and direction. Duration questions work because entry timestamps give the agent enough information to reason about start and end of activities. And retrieval itself is more precise because 70 distinct entries means every top-K result is relevant — no dilution from 40 identical hallway captions competing for the same slots.
