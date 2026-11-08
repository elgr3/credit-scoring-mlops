"""Contrats d'entrée / sortie de l'API (pydantic v2)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CheckingStatus = Literal["<0", "0<=X<200", ">=200", "no checking"]
CreditHistory = Literal[
    "no credits/all paid",
    "all paid",
    "existing paid",
    "delayed previously",
    "critical/other existing credit",
]
Purpose = Literal[
    "new car",
    "used car",
    "furniture/equipment",
    "radio/tv",
    "domestic appliance",
    "repairs",
    "education",
    "retraining",
    "business",
    "other",
]
SavingsStatus = Literal["<100", "100<=X<500", "500<=X<1000", ">=1000", "no known savings"]
Employment = Literal["unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"]
# "female single" et "vacation" existent dans la doc du jeu mais n'apparaissent dans aucune
# ligne : on ne les accepte pas, le modèle ne les a jamais vus.
PersonalStatus = Literal["male div/sep", "female div/dep/mar", "male single", "male mar/wid"]
OtherParties = Literal["none", "co applicant", "guarantor"]
PropertyMagnitude = Literal["real estate", "life insurance", "car", "no known property"]
OtherPaymentPlans = Literal["bank", "stores", "none"]
Housing = Literal["rent", "own", "for free"]
Job = Literal[
    "unemp/unskilled non res", "unskilled resident", "skilled", "high qualif/self emp/mgmt"
]
YesNone = Literal["none", "yes"]
YesNo = Literal["yes", "no"]

EXAMPLE_APPLICANT = {
    "checking_status": "<0",
    "duration": 24,
    "credit_history": "existing paid",
    "purpose": "radio/tv",
    "credit_amount": 3500,
    "savings_status": "<100",
    "employment": "1<=X<4",
    "installment_commitment": 3,
    "personal_status": "male single",
    "other_parties": "none",
    "residence_since": 2,
    "property_magnitude": "car",
    "age": 35,
    "other_payment_plans": "none",
    "housing": "own",
    "existing_credits": 1,
    "job": "skilled",
    "num_dependents": 1,
    "own_telephone": "none",
    "foreign_worker": "yes",
}


class CreditApplication(BaseModel):
    """Un dossier de demande de crédit (20 variables du German Credit)."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": EXAMPLE_APPLICANT})

    checking_status: CheckingStatus
    duration: int = Field(..., ge=1, le=120, description="Durée du crédit en mois")
    credit_history: CreditHistory
    purpose: Purpose
    credit_amount: float = Field(..., gt=0, le=1_000_000, description="Montant demandé (DM)")
    savings_status: SavingsStatus
    employment: Employment
    installment_commitment: int = Field(..., ge=1, le=4, description="Taux d'effort (1-4)")
    personal_status: PersonalStatus
    other_parties: OtherParties
    residence_since: int = Field(..., ge=1, le=4)
    property_magnitude: PropertyMagnitude
    age: int = Field(..., ge=18, le=100)
    other_payment_plans: OtherPaymentPlans
    housing: Housing
    existing_credits: int = Field(..., ge=1, le=10)
    job: Job
    num_dependents: int = Field(..., ge=0, le=10)
    own_telephone: YesNone
    foreign_worker: YesNo


class Prediction(BaseModel):
    probability_default: float = Field(..., ge=0, le=1)
    decision: Literal["approve", "reject"]
    threshold: float
    model_version: str


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    applications: list[CreditApplication] = Field(..., min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    predictions: list[Prediction]
    count: int
    model_version: str


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None
