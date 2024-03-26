from transformers import pipeline

custom_text = """
DevOps Engineer AiDash [ VC Funded ]
 Responsibilities include CI/CD Cloud infrastructure set up, SOC2 compliance readiness using Vanta.
● Executed a variety of tasks on the large number of machines of AWS services like EC2, S3, Workspaces, Lambda, VPC, IAM, Single Sign-On, RDS, Route53, Cloudwatch, Cloudfront using Python scripting via Boto3.
● Automated the creation of new tenants in Auth0 using Selenium
● Resolved daily ad-hoc tasks from various departments regarding malfunction of
services like Jenkins, Git, React etc, and fulfilling their requirements regarding AWS resources like EC2 instances, RDS, workspaces, S3 buckets etc, taking cost optimization standards under consideration.
● Used Datadog to create a variety of Synthetic tests, Dashboards, Monitors for all Aidash clients` applications health check up status.
● Solely executed Proof of Concept of shifting whole Aidash authentication and authorization system from Auth0 to Amazon Cognito, involving Username password authentication, SAML integration, Active directory integration, Support for custom domain, Email template support.
● Created Jenkins jobs to automate daily repeating ad-hoc tasks and automated AMI backup for Jenkins master and its slave
"""


model_name = "bayesian4042/bert_finetuned_resume"
classifier = pipeline("text-classification", model=model_name)
preds = classifier(custom_text, return_all_scores=True)
print(preds)