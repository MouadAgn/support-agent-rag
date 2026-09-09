"""Genere un PDF d'exemple (conditions de vente) a ingerer dans le RAG.

Cela montre que l'agent ne se limite pas a la FAQ : il exploite aussi des
documents PDF (garantie, retours, donnees personnelles...).

    python scripts/build_sample_pdf.py
"""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "pdf" / "conditions_generales.pdf"

SECTIONS = [
    (
        "Politique de garantie",
        "Tous les produits neufs beneficient d'une garantie legale de conformite "
        "de 2 ans a compter de la date de reception. La garantie couvre les "
        "defauts de fabrication mais pas l'usure normale ni les dommages "
        "resultant d'une mauvaise utilisation. Pour faire jouer la garantie, "
        "conservez votre preuve d'achat et contactez le support.",
    ),
    (
        "Produit defectueux a la reception",
        "Si un produit arrive endommage ou defectueux, signalez-le sous 14 jours. "
        "Nous organisons un retour gratuit et proposons au choix un echange ou un "
        "remboursement integral, frais de port inclus. Envoyez si possible une "
        "photo du produit et de l'emballage pour accelerer le traitement.",
    ),
    (
        "Remboursements",
        "Les remboursements sont effectues sur le moyen de paiement d'origine "
        "sous 14 jours apres validation du retour. Pour un achat regle en "
        "plusieurs fois, les echeances restantes sont annulees et les sommes "
        "deja prelevees sont recreditees.",
    ),
    (
        "Protection des donnees personnelles",
        "Vos donnees sont traitees conformement au RGPD. Vous disposez d'un droit "
        "d'acces, de rectification et de suppression de vos donnees. Pour exercer "
        "ces droits, ecrivez a privacy@boutique.example. Nous ne revendons "
        "jamais vos donnees a des tiers.",
    ),
    (
        "Programme de fidelite",
        "Chaque euro depense rapporte 1 point. 100 points donnent droit a un bon "
        "de 5 EUR utilisable des la commande suivante. Les points sont valables "
        "12 mois et visibles dans votre compte, rubrique Fidelite.",
    ),
]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUT), pagesize=A4,
                            leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()
    h = ParagraphStyle("h", parent=styles["Heading2"], spaceBefore=10, spaceAfter=6)
    body = ParagraphStyle("b", parent=styles["BodyText"], leading=15)

    flow = [Paragraph("Conditions generales de vente et de service", styles["Title"]),
            Spacer(1, 0.4 * cm)]
    for title, text in SECTIONS:
        flow.append(Paragraph(title, h))
        flow.append(Paragraph(text, body))
        flow.append(Spacer(1, 0.2 * cm))
    doc.build(flow)
    print(f"PDF genere : {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
