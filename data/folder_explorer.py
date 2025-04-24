import os

# Original folder structure (simplified)
folder_structure = {
    "State - People - Authorities": [
        "Federal Constitution",
        "Coat of Arms. Federal Seat. National Holiday",
        "Security of the Confederation",
        "Confederation and Cantons",
        "Citizenship. Settlement. Residence",
        "Fundamental Rights",
        "Political Rights",
        "Federal Authorities",
        "Foreign Affairs"
    ],
    "Private Law - Civil Justice - Enforcement": [
        "Civil Code",
        "Code of Obligations",
        "Intellectual Property and Data Protection",
        "Unfair Competition",
        "Cartels",
        "Civil Procedure",
        "Debt Enforcement and Bankruptcy",
        "Private International Law"
    ],
    "Criminal Law - Criminal Justice - Penal System": [
        "Civil Criminal Law",
        "Military Criminal Law",
        "Criminal Records",
        "Penal System",
        "Legal Assistance. Extradition",
        "Police Coordination and Services",
        "Helpers of Refugees during the Nazi Era"
    ],
    "Education - Science - Culture": [
        "Schools",
        "Science and Research",
        "Documentation",
        "Language. Arts. Culture",
        "Protection of Nature, Landscape and Animals"
    ],
    "National Defense": [
        "General National Defense",
        "Military Defense",
        "Civil Protection",
        "Economic Supply for National Defense"
    ],
    "Finance": [
        "General Organization",
        "Customs",
        "Taxes",
        "Information Exchange in Tax Matters",
        "Substitute for Military Service",
        "Exclusion from Tax Agreements. Double Taxation",
        "Alcohol Monopoly"
    ],
    "Public Works - Energy - Transport": [
        "National, Regional and Local Planning",
        "Expropriation",
        "Public Works",
        "Energy",
        "Transport",
        "Postal and Telecommunications"
    ],
    "Health - Labor - Social Security": [
        "Health",
        "Labor",
        "Social Insurance",
        "Housing",
        "Social Welfare",
        "Family Protection"
    ],
    "Economy - Technical Cooperation": [
        "Regional Policy",
        "Agriculture",
        "Forestry. Hunting. Fishing",
        "Industry and Commerce",
        "Trade",
        "Credit",
        "Insurance",
        "International Economic and Technical Cooperation",
        "Compensation of Swiss Interests"
    ]
}

# Save to file in tree format
def save_structure_to_file(structure, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for main_dir, sub_dirs in structure.items():
            f.write(f"[DIR] {main_dir}\n")
            for sub in sub_dirs:
                f.write(f"    [DIR] {sub}\n")
    print(f"Structure saved to {filename}")

# Example usage
save_structure_to_file(folder_structure, "translated_folder_structure.txt")

