# Ontology Repository

Welcome to the Ontology Repository! This repository contains a collection of ontologies designed to facilitate structured and standardized representation of knowledge in HyperLedger Fabric. These ontologies aim to improve their respective fields' data integration, knowledge sharing, and reasoning capabilities.

## Table of Contents

1. [Introduction](#introduction)
2. [Ontologies](#ontologies)
3. [Usage](#usage)

## Introduction

An ontology is a formal representation of knowledge in HyperLedger Fabric activities, often depicted as a set of concepts, their properties, and the relationships between them. This repository is a central hub for sharing ontologies created to understand specific domains better and enable more efficient data management and analysis.

## Ontologies

### 1. Activity Ontology

- **Description**: The "Activity Ontology" provides a comprehensive framework for representing hierarchically composed activities of HLF, including those with delays, whether organized in parallel or sequentially.

### 2. CSV Ontology

- **Description**: The "CSV Ontology" is a modelling framework for CSV (Comma-Separated Values) files and their constituent columns, designed to represent specific observation sets' structure and data types. This ontology incorporates essential object properties such as "hasValue" and "hasColumn," establishing critical links between the data values and columns within CSV files and concepts in other ontologies and their respective instances.

### 3. EDA Ontology

- **Description**: "Visualization Ontology" introduces novel classes representing different visualization types, including histograms, scatter plots, box plots, bar charts, and more. Additionally, this ontology introduces new object properties, such as "hasPlot," "hasX-Axis," and "hasY-Axis," designed to establish meaningful connections between visualizations and other ontologies.

### 4. Fabric Temporal Ontology

- **Description**: The "Fabric Temporal Ontology" handles time-related information within HLF activities. It defines concepts for activities' time aspects, like start times, durations, and end times. This ontology also introduces beneficial connections as object properties, such as "hasDuration," "hasStart," "hasEnd," and "hasTemporalData," which allow us to link time information with the activities, making it easier to understand how time factors into our data.

### 5. Fabric Deployment Ontology

- **Description**: The "Fabric Deployment Ontology" integrates the previous ontologies, encompassing a case-specific layer in our knowledge base, primarily focusing on individuals. We create a detailed picture of the HLF activities involved, including their timing, which we connect to specific columns in CSV files. We also visually represent these activities, helping us see how they connect and work with the data. 


## Usage

To use any of the ontologies in this repository, follow these steps:

1. Clone the repository to your local machine.
2. Navigate to the specific ontology's directory.
3. Review the ontology documentation to understand its structure, concepts, and relationships.
4. Import the ontology into your project using the appropriate ontology application (e.g. Protege).
5. Explore sample data and examples to see how the ontology can be applied in practice.




Thank you for visiting this Ontology Repository! We hope these ontologies are helpful for your projects. If you have any questions or suggestions, please don't hesitate to open an issue or contact us.
