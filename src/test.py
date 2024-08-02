test = {
        "question": "What is the primary purpose of a nucleus estate?",
        "answer": " \nBased on the retrieved context from the vector database, the primary purpose of a nucleus estate is to provide a structured and organized way for farmers to produce and market their agricultural products. Contract farming is a business model that involves a buyer and farmers entering into an agreement that establishes conditions for the production and marketing of a farm product or products. The primary purpose of a nucleus estate in this context is to provide a framework for farmers to produce high-quality products that meet the standards of the buyer, while also ensuring that the farmers are fairly compensated for their products.\nThe nucleus estate provides a number of benefits to farmers, including:\n1. Assured market: The nucleus estate provides a guaranteed market for farmers to sell their products, which can help them to stabilize their income and reduce the risks associated with crop failure or fluctuations in market prices.\n2. High-quality products: The nucleus estate provides a structured framework for farmers to produce high-quality products that meet the standards of the buyer. This can help to increase the value of the products and provide a competitive advantage in the market.\n3. Technical",
        "question_context": "The nucleus estate usually guarantees supplies to assure cost-efficient utilisation of installed processing capacities and to satisfy firm sales obligations respectively.\nIn some cases, the nucleus estate is used for research, breeding or piloting and demonstration purposes and/ or as collection point.\nThe farmers are at times called \u2018satellite farmers\u2019 illustrating their link to the nucleus farm. This model was in the past often used for state owned farms that re-allocated land to former workers. It is nowadays also used by the private sector as one type of CF. This model is often referred to as \u201coutgrower model\u201d.\nTypical products: perennials\nAdvantages\nContract farming is looking towards the benefits both for the farm-producers as well as to the agro-processing firms. Producer/farmer",
        "answer_context": "Key minimum requirements for appropriate contract schemes\nBroadly, the project must:\n\nContract Farming\n\nContents\nWhat is contract farming\nContract farming business models\nAdvantages\nChallenges\nPolicy support\nAgricultural produce suitable for CF\nKey minimum requirements for appropriate contract schemes\nRelated resources\nWhat is contract farming\nContract farming can be defined as agricultural production carried out according to an agreement between a buyer and farmers, which establishes conditions for the production and marketing of a farm product or products. Typically, the farmer agrees to provide agreed quantities of a specific agricultural product. These should meet the quality standards of the purchaser and be supplied at the time determined by the purchaser. In turn, the buyer commits to purchase the product and, in some cases, to support production through, for example, the supply of farm inputs, land preparation and the provision of technical advice.\n\nContract farming business models\nInformal model - This model is the most transient and speculative of all contract farming models, with a risk of default by both the promoter and the farmer\u201d (van Gent, n.d., p.5). However, this depends on the situation: interdependence of contract parties or long-term trustful relationships may reduce the risk of opportunistic behaviour. Special features of this CF model are:\nSmall firms conclude simple, informal seasonal production contracts with smallholders.\nThe success often depends on the availability and quality of external extension services.\nEmbedded services, if at all provided, are limited to the delivery of basic inputs, occasionally on credit; advice is usually limited to grading and quality control.\nTypical products: requiring minimal processing/ packaging, vertical coordination; e.g. fresh fruit/ vegetables for local markets, sometimes also staple crops.\n\nhave clear mechanisms for settling disputes.\nThe government should:\nact as a third party, or mediator, between the parties and not be a mouthpiece for the company sponsor;\nhave appropriate legislation to ensure that farmers\u2019 rights can be enforced.\nRelated resources\nAn analytical toolkit for support to Contract farming\nLegal guide on Contract farming\nContract farming and out-grower schemes - Appropriate development models to tackle poverty and hunger?",
        "coverage": {
            "score": 4,
            "reasoning": "The answer can be partially extracted from the provided context, as it summarizes some of the benefits of a nucleus estate. However, it also contains information that is not explicitly mentioned in the context, such as the framework for farmers to produce high-quality products and meet buyer standards. Nevertheless, the answer stays within the general topic and does not hallucinate or introduce unrelated concepts."
        },
        "relevancy": {
            "score": 4,
            "reasoning": "The answer partially addresses the main aspects of the question. It correctly identifies that a nucleus estate provides a structured and organized way for farmers to produce and market their agricultural products, but it does not explicitly mention the benefits of assured market, high-quality products, and technical support mentioned in the context. The answer also mentions some benefits of contract farming, which is related to the topic of nucleus estates. However, the answer does not directly address the primary purpose of a nucleus estate."
        },
        "groundedness": {
            "properties": {
                "score": {
                    "maximum": 5,
                    "minimum": 1,
                    "title": "Score",
                    "type": "integer"
                },
                "reasoning": {
                    "title": "Reasoning",
                    "type": "string"
                }
            },
            "required": [
                "score",
                "reasoning"
            ]
        }
    }

print(test['groundedness'].keys())