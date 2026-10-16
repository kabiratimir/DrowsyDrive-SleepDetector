# Project Notes

## Project title

Student Sleep Detection & Alert System Using Python and Computer Vision

## Core idea

Webcam -> facial landmarks -> eye landmarks -> EAR -> temporal analysis -> drowsiness/sleep state -> timer/alarm -> session statistics.

## Important terminology

**Drowsiness:** A state indicating reduced alertness. In this project it is inferred from visual cues.

**Probable sleep:** Prolonged eye closure satisfying configured rules. This is not a medical diagnosis.

**EAR:** Eye Aspect Ratio, a geometric measure based on distances between selected eye landmarks.

## Why time matters

A single closed-eye frame can be a blink. A robust system must consider how long the eyes remain closed and should use smoothing/state logic to reduce false alarms.
