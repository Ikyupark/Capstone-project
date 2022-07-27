-- Exported from QuickDBD: https://www.quickdatabasediagrams.com/
-- NOTE! If you have used non-SQL datatypes in your design, you will have to change these here.


CREATE TABLE "fake_job_postings" (
    "job_id" INT   NOT NULL,
    "title" VARCHAR,
    "location" VARCHAR,
    "department" VARCHAR,
    "salary_range" VARCHAR,
    "company_profile" VARCHAR,
    "description" VARCHAR,
    "requirements" VARCHAR,
    "benefits" VARCHAR,
    "telecommuting" INT,
    "has_company_logo" INT,
    "has_questions" INT,
    "employment_type" VARCHAR,
    "required_experience" VARCHAR,
    "required_education" VARCHAR,
    "industry" VARCHAR,
    "function" VARCHAR,
    "fraudulent" INT,
    CONSTRAINT "pk_fake_job_postings" PRIMARY KEY (
        "job_id"
     )
);

