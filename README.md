Using OpenShift 4.20.6 and OpenShift AI 3.2.0.

### Log in OpenShift

```bash
oc login ...
```

### Make sure that at least a GPU enabled worked has this label:

```yaml
group: llama-stack-demo
```

### Create the project

```bash
PROJECT="llama-stack-demo"

oc new-project ${PROJECT}
``` 

Label project with
- modelmesh-enabled: 'false'
- opendatahub.io/dashboard: 'true'

```bash
oc label namespace ${PROJECT} modelmesh-enabled=false opendatahub.io/dashboard=true
```

Install the Grafana operator.

### Install with Helm

This default deployment deploys one model.

```bash

helm install llama-stack-demo helm/ --namespace ${PROJECT}  --timeout 20m
```

# load your .env locally
set -a
source .env
set +a

# create secret in cluster
oc -n llama-stack-demo create secret generic tavily-key \
  --from-literal=TAVILY_SEARCH_API_KEY="$TAVILY_API_KEY"

### Wait for pods

```bash
oc -n ${PROJECT} get pods -w
```

Expected pods (may take 5-10 minutes to start):
```
(Output)
NAME                                              READY   STATUS    RESTARTS   AGE
eligibility-lsd-0                                1/1     Running   0          8m
eligibility-lsd-playground-0                     1/1     Running   0          8m
eligibility-engine-0                             1/1     Running   0          7m
loader-0                                         0/1     Completed 0          6m
llama-3-1-8b-w4a16-predictor-df76b56d6-fw8fp    2/2     Running   0          10m
```

### Test

You can access the system in multiple ways:

#### Option 1: OpenShift AI Dashboard
Get the OpenShift AI Dashboard URL:
```bash
oc get routes rhods-dashboard -n redhat-ods-applications
```

Navigate to Data Science Projects -> llama-stack-demo. You'll see the deployed models and workbenches.

#### Option 2: Direct Access to Llama Stack Playground
Get the Llama Stack Playground URL:
```bash
oc get routes eligibility-lsd-playground -n ${PROJECT}
```

Access the playground directly to interact with the eligibility assessment system.

#### Option 3: API Access
For programmatic access, get the Llama Stack API endpoint:
```bash
oc get routes eligibility-lsd -n ${PROJECT}
```

Use this endpoint to integrate the eligibility assessment capabilities into your applications.

## Business Rules

### Unpaid Leave Evaluation Data

| Family relationship | Situation | Single-parent family | Number of children | Potentially eligible | Monthly benefit | Case | Description | Output | DESCRIPTION | Rule ID |
|---------------------|-----------|---------------------|-------------------|---------------------|----------------|------|-------------|--------|-------------|---------|
| true | delivery, birth | true | | true | 500 | E | Single-parent family with newborn | The single-parent status must be documented | Case E: Single-parent family with any child | regla-005 |
| true | delivery, birth | | >=3 | true | 500 | B | Third child or more with newborn | The number of children must be 3 or more, the ages of at least 2 of the minors must be less than 6, if there is disability greater than 33% then the limit is 9 years | Case B: Third child or more with newborn | regla-002 |
| true | delivery, birth | | | false | 0 | B | The number of children must be 3 or more, must consult with administration | | The number of children must be 3 or more, must consult with administration | 9ec43eb2-484f-4fcf-9dd7-6510da30850c |
| true | illness, accident | | | true | 725 | A | First-degree family care sick or accident victim | The person must have been hospitalized and the care of the person must be continued | Case A: First-degree family care sick/injured | regla-001 |
| true | adoption, foster_care | | | true | 500 | C | Adoption or foster care | In the foster care case the duration must be longer than one year | Case C: Adoption or foster care | regla-003 |
| true | multiple_birth, multiple_delivery, multiple_adoption, multiple_foster_care | | | true | 500 | D | Delivery, adoption or foster care multiple | | Case D: Delivery, adoption or foster care multiple | regla-004 |
| true | | | | false | 0 | NONE | No case applies | | No case applies | 515afd1f-43cc-44ed-971c-fefb273840b2 |
| false | | | | false | 0 | NONE | Not applicable by relationship (first degree) | | Only father, mother, son, daughter, spouse or partner are accepted | 058dd988-90dd-46da-8478-ee458aacde6f |
| | | | | false | 0 | NONE | UNKNOWN_ERROR | | | f32bfb0f-801d-4d6c-b5bd-13a1edd0eaca |

### Summary

This table contains the evaluation criteria and outcomes for unpaid leave assistance eligibility. The data shows different cases (A through E) with varying monthly benefits:

- **Case A**: First-degree family care sick/injured - 725€
- **Case B**: Third child or more with newborn - 500€  
- **Case C**: Adoption or foster care - 500€
- **Case D**: Multiple delivery/adoption/foster care - 500€
- **Case E**: Single-parent family with any child - 500€
- **NONE**: Cases where no assistance applies - 0€

The table includes input parameters (family relationship, situation, single-parent status, number of children) and corresponding outputs (eligibility, benefit amount, case classification, descriptions, and rule IDs).

## Example queries

- My mother had an accident and she's at the hospital. I have to take care of her, can I get access to the unpaid leave aid?
- My mother had an accident and she's at the hospital. I have to take care of her, tell me if I can get access to the unpaid leave aid and the requirements I have to meet.
- I have just adopted two children, at the same time, aged 3 and 5, am I elegible for the unpaid leave aid? How much?
- I have just adopted two children, at the same time, aged 3 and 5, tell me if I'm elegible for the unpaid leave aid and which requirements I should meet.
- I'm a single mom and I just had a baby, may I get access to the unpaid leave aid?
- Enumerate the legal requirements to get the aid for unpaid leave.

## Example System Prompt

You are a helpful AI assistant that uses tools to help citizens of the Republic of Lysmark. Answers should be concise and human readable. AVOID references to tools or function calling nor show any JSON. Infer parameters for function calls or instead use default values or request the needed information from the user. Call the RAG tool first if unsure. Parameter single_parent_family only is necessary if birth/adoption/foster_care otherwise use false.

## Uninstall

Unistall the helm chart.

```bash
helm uninstall llama-stack-demo --namespace ${PROJECT}
```

Delete all remaining objects like jobs created in hooks.

```bash
oc delete jobs -l "app.kubernetes.io/part-of=llama-stack-demo"
```

Finally remove the project:

```bash
oc delete project ${PROJECT}
```

# Monitoring

```yaml
apiVersion: dscinitialization.opendatahub.io/v2
kind: DSCInitialization
metadata:
  name: default-dsci
spec:
  applicationsNamespace: redhat-ods-applications
  monitoring:
    alerting: {}
    managementState: Managed
    metrics:
      replicas: 1
      storage:
        retention: 90d
        size: 50Gi
    namespace: redhat-ods-monitoring
    traces:
      sampleRatio: '1.0'
      storage:
        backend: pv
        retention: 2160h0m0s
        size: 100Gi
  trustedCABundle:
    customCABundle: ''
    managementState: Managed
```

Just make sure that in your default-dsci spec->monitoring is coherent with:

```yaml
spec:
  ...
  monitoring:
    alerting: {}
    managementState: Managed
    metrics:
      replicas: 1
      storage:
        retention: 90d
        size: 50Gi
    namespace: redhat-ods-monitoring
    traces:
      sampleRatio: '1.0'
      storage:
        backend: pv
        retention: 2160h0m0s
        size: 100Gi
...
```

### BUG collector service

Create this service...

```yaml
kind: Service
apiVersion: v1
metadata:
  name: data-science-collector
  namespace: redhat-ods-monitoring
  labels:
    app.kubernetes.io/component: opentelemetry-collector
    app.kubernetes.io/instance: redhat-ods-monitoring.data-science-collector
    app.kubernetes.io/part-of: opentelemetry
spec:
  ipFamilies:
    - IPv4
  ports:
    - name: otlp-grpc
      protocol: TCP
      appProtocol: grpc
      port: 4317
      targetPort: 4317
    - name: otlp-http
      protocol: TCP
      appProtocol: http
      port: 4318
      targetPort: 4318
    - name: prometheus
      protocol: TCP
      port: 8889
      targetPort: 8889
  internalTrafficPolicy: Cluster

  type: ClusterIP
  ipFamilyPolicy: SingleStack
  sessionAffinity: None
  selector:
    app.kubernetes.io/component: opentelemetry-collector
    app.kubernetes.io/instance: redhat-ods-monitoring.data-science-collector
    app.kubernetes.io/managed-by: opentelemetry-operator
    app.kubernetes.io/part-of: opentelemetry
```