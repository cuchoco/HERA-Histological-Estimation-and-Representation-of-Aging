"""Constants shared by training, fusion and inference."""
import re

# Age is standardized with the mean / s.d. of the GTEx cohort (25,702 slides).
AGE_MEAN = 53.830
AGE_STD = 12.489

FEATURE_DIM = 1536          # UNI2-h patch embedding

# 40 GTEx tissue types -> 29 organs
TISSUE_TO_ORGAN = {
    'Adipose_Subcutaneous': 'Adipose',
    'Adipose_Visceral_Omentum': 'Adipose',
    'Adrenal_Gland': 'Adrenal_Gland',
    'Artery_Aorta': 'Artery',
    'Artery_Coronary': 'Artery',
    'Artery_Tibial': 'Artery',
    'Bladder': 'Bladder',
    'Brain_Cerebellum': 'Brain',
    'Brain_Cortex': 'Brain',
    'Breast_Mammary_Tissue': 'Breast',
    'Cervix_Ectocervix': 'Cervix',
    'Cervix_Endocervix': 'Cervix',
    'Colon_Sigmoid': 'Colon',
    'Colon_Transverse': 'Colon',
    'Esophagus_Gastroesophageal_Junction': 'Esophagus',
    'Esophagus_Mucosa': 'Esophagus',
    'Esophagus_Muscularis': 'Esophagus',
    'Fallopian_Tube': 'Fallopian_Tube',
    'Heart_Atrial_Appendage': 'Heart',
    'Heart_Left_Ventricle': 'Heart',
    'Kidney_Cortex': 'Kidney',
    'Kidney_Medulla': 'Kidney',
    'Liver': 'Liver',
    'Lung': 'Lung',
    'Minor_Salivary_Gland': 'Minor_Salivary_Gland',
    'Muscle_Skeletal': 'Muscle',
    'Nerve_Tibial': 'Nerve',
    'Ovary': 'Ovary',
    'Pancreas': 'Pancreas',
    'Pituitary': 'Pituitary',
    'Prostate': 'Prostate',
    'Skin_Not_Sun_Exposed_Suprapubic': 'Skin',
    'Skin_Sun_Exposed_Lower_leg': 'Skin',
    'Small_Intestine_Terminal_Ileum': 'Small_Intestine',
    'Spleen': 'Spleen',
    'Stomach': 'Stomach',
    'Testis': 'Testis',
    'Thyroid': 'Thyroid',
    'Uterus': 'Uterus',
    'Vagina': 'Vagina',
}

# The 24 organs with > 250 slides get their own per-organ clock. The other five
# (Minor_Salivary_Gland, Uterus, Bladder, Cervix, Fallopian_Tube) use the pan-organ clock only.
PER_ORGAN_CLOCKS = [
    'Esophagus', 'Artery', 'Skin', 'Adipose', 'Colon', 'Heart', 'Muscle', 'Nerve',
    'Stomach', 'Thyroid', 'Breast', 'Spleen', 'Pancreas', 'Lung', 'Brain',
    'Small_Intestine', 'Adrenal_Gland', 'Liver', 'Kidney', 'Prostate', 'Testis',
    'Pituitary', 'Vagina', 'Ovary',
]


def clean_tissue(name: str) -> str:
    """'Skin - Sun Exposed (Lower leg)' -> 'Skin_Sun_Exposed_Lower_leg' (GTEx SMTSD -> our key)."""
    return re.sub(r'[^a-zA-Z0-9]+', '_', str(name)).strip('_')
