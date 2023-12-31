# Instruction Tuning Update for CodeT5 

## Overview
This document outlines the challenges encountered during the Instruction-Tuning process of CodeT5, primarily due to computational limitations. The testing-phase of CodeT5 has give good results and one of the results is available in Testing_CodeT5_Results folder here.

## Issue Description 📝
- The Instruction-Tuning of CodeT5 failed because of insufficient computation resources during the sharding process. The primary constraint was limited memory availability.
- Attempts to run the training code on Google Colab+ (which offers 40 GB of GPU and additional memory) were unsuccessful, indicating that the computational demands exceed even enhanced cloud computing capabilities.
  
![Image](Extra/Sharding_Failure.png)


## Current Actions and Future Plans 🔜
- Professor Zhe Yu and I (John Melwin Richard) have requested access to RIT-Sporc, a more powerful computing resource.
- Once access is granted, we plan to resume the training process on RIT-Sporc, anticipating that its superior computational capabilities will overcome the current limitations.
![Alt text](Extra/Request_SPORC.png)

---

This README provides a brief on the encountered issues and the steps being taken to resolve them, ensuring continued progress in our project.

