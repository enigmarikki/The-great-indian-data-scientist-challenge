# The-great-indian-data-scientist-challenge

## The Problem Statement:
Organizations around the world purchase goods & services from their suppliers via Purchase Order-Invoice exchange. Organizations raise Purchase Order for specific item and expected price. Suppliers then raise an Invoice to the organization for billing. As part of process streamlining, organizations catalogue items, i.e., they document all the item details in their procurement system. However in many cases organizations are required to purchase non-catalogued items.
For any kind of analysis of spend data, it is important that all invoices be classified to specific “Product Category”, defined by a standard taxonomy that the organization follows.
Example, an invoice having an item description of “Mobile Bills” may be classified into “Telecom” product category.

## Data Description:
The given csv file had the following description
|Column|Expansion|Description|
|------|------|------|
|Inv_ID|Invoice ID|Unique number representing Invoice created by supplier/vendor|
|Vendor Code|Vendor ID|Unique number representing Vendor/Seller in the procurement system|
|GL_Code|Account’s Reference ID| 
|Inv_Amt|Invoice Amount| 
|Item Description|Description of Item Purchased|Example: “Corporate Services Human Resources Contingent Labor/Temp Labor Contingent Labor/Temp Labor”|
|Product Category|Category of Product for which Invoice is raised|A pseudo product category is represented in the dataset as CLASS-???, where ? is a digit.|

## Submission Format:
Please submit the prediction as a .csv file in the format described below and in the sample submission file.

|Inv_Id|Product_Category|
|------|------|
|1|CLASS-784|
|2|CLASS-784|
|3|CLASS-784|
|4|CLASS-784|
|5|CLASS-784|
|6|CLASS-784|

## Results
The model based on Random Forest Tree had an overall score of 99.80%
