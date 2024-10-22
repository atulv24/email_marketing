import os
import re
import csv
import json
import time
import email
import openai
import imaplib
import smtplib
import operator
from bs4 import BeautifulSoup
from email.mime.text import MIMEText
from email.header import decode_header
from email.mime.multipart import MIMEMultipart
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import END
from langgraph.graph import StateGraph
from typing import Annotated, TypedDict
from typing import TypedDict, Annotated, List, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
from datetime import datetime
from datetime import datetime, timedelta
import imaplib
import email
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
import streamlit as st
import random

os.environ["OPENAI_API_KEY"] = "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"
sender_email= "Atul.v@factspan.com"
password = "sieptuaafdkhccqc"
smtp_server = 'smtp.gmail.com'
smtp_port = 587
imap_server = 'imap.gmail.com'
imap_port = 993
file_name = "./sample_data.csv"


lead_list = [
    {
        "Company Name": "Factspan",
        "Contact Name": "Atul Verma",
        "Job Title": "SPA ",
        "Email": "atul.verma@factspan.com",
        "Phone Number": "+1 123 456 7890",
        "Industry": "Software",
        "Company Size": "51-200 employees",
        "Location": "California, USA",

    },
    {
        "Company Name": "Factspan",
        "Contact Name": "Charan Reddy",
        "Job Title": "Senior Analyst",
        "Email": "charan.reddy@factspan.com",
        "Phone Number": "+1 123 456 7890",
        "Industry": "Software",
        "Company Size": "51-200 employees",
        "Location": "Bangalore, IN",
    }   
]


st.title("Inside Sales Executive Agent")

# Assuming you have sender_email and password defined somewhere above

#if 'input' not in st.session_state:
#st.session_state['input'] = {}
# Initialize session state for input if it doesn't exist
if 'input' not in st.session_state:
    st.session_state.input = {
        "Company Name": "Factspan",
        "Contact Name": "Vikas Chavan",
        "Job Title": "Director",
        "Email": "vikas.chavan@factspan.com",
        "Phone Number": "+1 123 456 7890",
        "Industry": "Software",
        "Company Size": "51-200 employees",
        "Location": "Bangalore, IN",
    }

# Create a form to collect user input
with st.sidebar.form("lead_form"):
    st.write("Enter Lead details:")
    company_name = st.text_input("Company Name", value=st.session_state.input["Company Name"], key="company_name")
    contact_name = st.text_input("Contact Name", value=st.session_state.input["Contact Name"], key="contact_name")
    job_title = st.text_input("Job Title", value=st.session_state.input["Job Title"], key="job_title")
    email_id = st.text_input("Email", value=st.session_state.input["Email"], key="email")
    # phone_number = st.text_input("Phone Number", value=st.session_state.input["Phone Number"], key="phone_number")
    # industry = st.text_input("Industry", value=st.session_state.input["Industry"], key="industry")
    # company_size = st.text_input("Company Size", value=st.session_state.input["Company Size"], key="company_size")
    location = st.text_input("Location", value=st.session_state.input["Location"], key="location")

    # Submit button
    submitted = st.form_submit_button("Submit")

    if submitted:
        # Update session state with the new input
        st.session_state.input = {
            "Company Name": company_name,
            "Contact Name": contact_name,
            "Job Title": job_title,
            "Email": email_id,
            "Phone Number": "0123456789",
            "Industry": "Software",
            "Company Size": "200-500 employees",
            "Location": location,
        }

        # Assuming lead_list is defined elsewhere in your app
        lead_list.append(st.session_state.input)
        st.success("Lead submitted successfully!")

def should_continue(state):
    
    receiver_email = st.session_state.input['Email']
    # Connect to the server
    try:
        with st.status("Checking for Emails...",expanded=True):
            mail = imaplib.IMAP4_SSL("imap.gmail.com")
            st.write("Login to Gmail...")
            # Login to your account
            mail.login(sender_email, password)
            # Select the mailbox you want to check
            mail.select('"[Gmail]/Sent Mail"')
            st.write("Searching for new messages...")
            # Search for the email
            status, messages = mail.search(None, 'TO', receiver_email)
            # Convert messages to a list of email IDs
            email_ids = messages[0].split()
        
        if email_ids:
            st.info(f"Email already sent to the recipient: {state['input']['Email']}")
            #st.info(f"Email already sent to the recipient:")
            return "conversation handler"
        else:
            st.info(f"Email sent to the recipient for the first time: {state['input']['Email']}")
            return "continue"
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        mail.logout() if 'mail' in locals() else None

def check_email_status(state):
    return state


def analyze_question(state):
    llm = ChatOpenAI(
    model = "gpt-4o-mini",temperature=0,streaming=True)
    prompt = PromptTemplate.from_template("""
    You are an agent that needs to define if a question is a sql code one or a general public one.

    Question : {input}

    Analyse the question. Only answer with "crm_integration_agent" if the question is about technical sql development. If not just answer "public_agent".

    Your answer (crm_integration_agent/public_agent) :
    """)
    chain = prompt | llm
    response = chain.invoke({"input": state["input"]})
    decision = response.content.strip().lower()
    #print(decision)
    return {"decision": decision, "input": state["input"]}

# Creating the code agent that could be way more technical
def crm_integration_agent(state):
    llm = ChatOpenAI(
    model = "gpt-4o-mini",temperature=0,streaming=True)
    prompt = PromptTemplate.from_template(
        "You are a SQL software engineer. Answer this question with step by steps details : {input}"
    )
    chain = prompt | llm
    response = chain.invoke({"input": state["input"]})
    return {"output": response}

# Creating the generic agent
def public_agent(state):
    with st.status("Fetching User Profile...",expanded=True):
        from openai import OpenAI
        YOUR_API_KEY = "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"
        company = state["input"]["Company Name"]
        name = state["input"]["Contact Name"]
        content = f"Provide all the information for {name} in {company} company?"
        messages = [
            {"role": "system", "content": "You are an AI assistant."},
            {"role": "user", "content": content}
        ]

        client = OpenAI(api_key=YOUR_API_KEY, base_url="https://api.perplexity.ai")
        response = client.chat.completions.create(
            model="llama-3.1-sonar-small-128k-online",
            #model="llama-3.1-sonar-small-128k-chat",
            messages=messages,
        )
        #print(response.choices[0].message.content)
        response = response.choices[0].message.content
    st.write(f"{name}'s Profile :",response)
    return {"output": response}

def customer_intelligence_report(state):
    input_context = state["output"]
    os.environ["OPENAI_API_KEY"] = "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"

    prompt = f"""

    You are an AI assistant tasked with helping a team decide which pitch to use for their presentation. The team has the following pitches available:

    1. Data Governance Pitch
    2. Health Organization and Healthcare Pitch
    3. Generative AI Prospects Pitch
    4. General Sales Pitch
    5. Product Analysis Pitch
    6. Artificial Intelligence Pitch
    7. Data Engineering Pitch
    8. Other Pitches

    Based on the input context provided, analyze the key points and suggest the most suitable pitch only.
    Input Context:
    {input_context}
    if information is there the Your answer (Data Governance Pitch/Health Organization and Healthcare Pitch/Generative AI Prospects Pitch/General Sales Pitch/Product Analysis Pitch/Artificial Intelligence Pitch/Data Engineering Pitch) and
    if no information is there then Your answer (Other Pitches) :

    """

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        streaming=True
    )

    response = llm.invoke(prompt)
    response = response.content.strip().lower()
    # st.write("customer intelligence report",response)
    #print(response)
    return {"output": response}

def compose_email(state):
    with st.status("Drafting the Email...",expanded=True):
        pitch_type = state["output"]
        company_name = state["input"]["Company Name"]
        recipient_name = state["input"]["Contact Name"]
        your_name = "Atul Verma"
        your_contact_info = "Atul.v@factspan.com"
        if pitch_type == "data_ai_governance":
            email_body = f"""
            Hi {recipient_name},

            I am fascinated by the Data & AI initiatives {company_name} is pioneering to enhance healthcare operations and patient experience.

            At Factspan, we share a similar passion for harnessing Data & AI to revolutionize healthcare. We have partnered with leading healthcare organizations like CVS, Baptist Health, and Rite Aid, enhancing their in-house capabilities to deliver superior patient care and operational efficiency.

            **Data & AI Governance services:** What sets us apart is our "Unified Intelligence Governance" framework. The approach seamlessly integrates traditional data governance with AI-specific requirements, ensuring a comprehensive governance structure. By addressing data, people, processes, and policies, alongside model ethics, lifecycle management, and compliance, we offer a holistic solution for managing your organization’s most valuable assets.
            - Establishing a robust governance foundation
            - Building an intelligence architecture
            - Identifying risks and opportunities
            - Protecting data subject rights & ethics management
            - Securing and optimizing data and AI models

            Our expertise in Data & AI spans across:
            - Data Engineering
            - Data Governance
            - Strategic Analytics
            - Data Science and ML
            - Generative AI

            As you continue to drive your data governance initiatives, I believe our expertise can serve as a valuable extension to your efforts.

            Would you be interested in exploring further? Can I schedule a brief call?

            Cheers,
            {your_name}
            Sr. Director of Marketing
            www.factspan.com
            e: {your_contact_info}
            """
        elif pitch_type == "health_organization_and_healthcare_pitch":
            email_body = f"""
            Hi {recipient_name},

            I am fascinated by the AI initiatives {company_name} is pioneering to enhance healthcare operations and patient experience.

            At Factspan, we share a similar passion for harnessing Data & AI to revolutionize healthcare. We have partnered with leading healthcare organizations like CVS, Baptist Health, and Rite Aid, enhancing their in-house capabilities to deliver superior patient care and operational efficiency.

            One of our flagship projects is a Persona-Based Real-Time Patient Insights Platform. This AI-driven platform provides contextual insights to doctors, nurses, and hospital staff, delivering real-time data on patient journeys and diagnoses at departmental, process, and individual patient levels. The outcomes have been transformative:
            Enhanced Patient Experience: Personalized insights improve patient interactions and satisfaction
            Optimized Staff Scheduling: Data-driven scheduling boosts staff productivity and reduces burnout.
            Operational Efficiency: Streamlined processes cut operational costs and improve system efficiency.
            What sets us apart in data governance is our "Unified Intelligence Governance" framework. This comprehensive approach seamlessly integrates traditional data governance with AI-specific requirements, addressing data, people, processes, and policies, along with model ethics, lifecycle management, and compliance. This ensures a robust structure for managing your organization’s most valuable assets.

            Our expertise in Data & AI spans across:
            Data Engineering
            Data Governance
            Strategic Analytics
            Data Science and ML
            Generative AI
            I would love to hear about your initiatives and discuss how we can support and amplify Envision Healthcare’s innovative efforts.

            Could we schedule a brief call to explore potential synergies?

            Cheers,
            {your_name}
            Sr. Director of Marketing
            www.factspan.com
            e: {your_contact_info}
            """
        elif pitch_type == "generative_ai_prospects_pitch":
            email_body = f"""
            Hi {recipient_name},

            I came across your impressive work leading Optum's Gen AI initiatives at {company_name} and felt compelled to reach out. Given the critical importance of AI in healthcare, I believe Factspan could offer significant support as you advance this initiative.

            At Factspan, we help industry leaders like CVS, Baptist health, and RiteAid to accelerate their data-to-decisions journey. Our expertise spans Data Engineering, Data Governance, Strategic Analytics, Data Science, and AI.

            Through our Gen AI capabilities, we enable physicians, insurers, hospital administrators, and pharmacists to gain precise, contextual insights. These insights encompass real-time demand prediction, staffing optimization, inventory scheduling, automated diagnostic report analysis, preventive healthcare, and more.

            We recently featured an episode on our "The Fluid Intelligence Podcast" channel, titled "Personalizing Patient Care Through Data and AI," with the CDO of Baptist Health. You can listen to it here, when you have time: [link to episode: https://www.youtube.com/watch?v=I2z70SAoW10].

            I believe our deep expertise and proven track record can serve as a valuable extension to your efforts in building advanced AI solutions.

            Would you like to see a demo of our AI capabilities for healthcare?

            Look forward to your response.


            Cheers,
            {your_name}
            Sr. Director of Marketing
            www.factspan.com
            e: {your_contact_info}
            """
        elif pitch_type == "product_analysis_pitch":
            email_body = f"""
            Hi {recipient_name},

            I recently saw job postings for Cloud Engineer, Data Engineering & Governance roles at {company_name} and felt compelled to reach out.

            At Factspan, we help industry leaders like Disney, Macy's, and CVS to accelerate their data-to-decisions journey. Our expertise in Data & AI spans Data Engineering, Data Governance, Strategic Analytics, Data Science, and AI.

            On the product analytics side, we do a lot of work around product performance analysis, pricing optimization, promotion & loyalty analysis, product adoption, and more. One of our recent work involved a multi-model approach to analyse the uploaded image and text of the product in an eCommerce site and auto recommend the tagging of product tags and attributes for better and relevant search visibility of the products to their customers.

            AI for production support: Our Integrated Command Center (ICC) provides a consolidated view of your production environment. From anticipating job failures to managing incidents, overseeing platform help and addressing service alarms, it acts as a comprehensive dashboard for real time insights. The Autoheal feature automatically corrects error when there is enough confidence level. https://www.youtube.com/watch?v=u_nPw49S2bg

            Would you like to see a demo of our Data & AI capabilities? Or could you refer to someone in your team who would be interested in exploring further?

            Look forward to your response.

            Cheers,
            {your_name}
            Sr. Director of Marketing
            www.factspan.com
            e: {your_contact_info}
            """
        elif pitch_type == "artificial_intelligence_pitch":
            email_body = f"""
            Hi {recipient_name},

            I am fascinated by the numerous Data & AI initiatives Hartford HealthCare is pioneering to enhance healthcare operations and patient experience. Congratulations on the successful launch of the Center for Artificial Intelligence (AI) Innovation in Healthcare.

            At Factspan, we share your passion for harnessing Data & AI to revolutionize healthcare. We have partnered with leading healthcare organizations like CVS, Baptist Health, and Rite Aid, enhancing their in-house capabilities to deliver superior patient care and operational efficiency.

            Persona-Based Real-Time Patient Insights Platform: One of our flagship projects is an AI platform designed for hospitals. This platform provides real-time, persona-based contextual insights to Doctors, Nurses, and Hospital staff, into patient journeys and diagnoses at departmental, process, and individual patient levels. The results have been transformative:
            Enhanced Patient Experience: Personalized insights improve patient interactions and satisfaction
            Optimized Staff Scheduling: Data-driven scheduling boosts staff productivity and reduces burnout.
            Operational Efficiency: Streamlined processes cut operational costs and improve system efficiency.
            AI governance framework: What sets us apart in data governance is our "Unified Intelligence Governance" framework. The approach seamlessly integrates traditional data governance with AI-specific requirements, ensuring a comprehensive governance structure. By addressing data, people, processes, and policies, alongside model ethics, lifecycle management, and compliance, we offer a holistic solution for managing your organization’s most valuable assets.

            Our expertise in Data & AI spans across:
            Data Engineering
            Data Governance
            Strategic Analytics
            Data Science and ML
            Generative AI
            I would love to explore how Factspan can support and extend Hartford HealthCare’s innovative efforts.

            Would you be open to scheduling a demo of our AI platform?


            Cheers,
            {your_name}
            Sr. Director of Marketing
            www.factspan.com
            e: {your_contact_info}
            """
        elif pitch_type == "data_engineering_pitch":
            email_body = f"""
            Hi {recipient_name},

            We are a Data and AI company that partners with healthcare leaders such as CVS, Baptist Health, and RiteAid to enhance their in-house capabilities across Data Engineering, Analytics, and AI.
            Our Data Engineering expertise helps healthcare providers like yours build robust, scalable data pipelines that streamline operations and enhance decision-making. We specialize in:
            •	Data Integration & Transformation: Seamlessly bringing together data from EHRs, claims systems, and operational systems into unified data lakes or warehouses.
            •	Data Quality & Governance: Implementing rigorous data validation and governance frameworks to ensure accuracy, compliance, and trust in your data.
            •	Cloud Architecture & Optimization: Designing and optimizing cloud-native data solutions that provide secure, efficient, and scalable access to critical healthcare data.
            •	Real-time Data Processing: Enabling real-time data flow for timely insights on patient outcomes, resource allocation, and operational efficiencies.
            •	Data Pipeline Automation: Automating repetitive data tasks, ensuring smooth data ingestion, and improving the speed of analytics delivery.
            Our solutions help healthcare organizations drive better decisions through data-driven insights, improve patient outcomes, and reduce operational inefficiencies. We have a track record of transforming data infrastructure to enable real-time analytics and support advanced AI solutions.
            I believe our deep Data Engineering expertise and proven results can significantly benefit UCLA’s Data and AI teams.
            Would you be open to a discussion or a demo of how we can support your healthcare data initiatives? Alternatively, could you connect me with the right person from your team who handles Data Engineering?
            Looking forward to hearing from you.

            Cheers,
            {your_name}
            Sr. Director of Marketing
            www.factspan.com
            e: {your_contact_info}
            """
        else:
            email_body = f"""
        Hi {recipient_name},

        We are a Data and AI company working with healthcare companies like CVS, Baptist Health, and RiteAid to extend their in-house team capabilities in Data,Analytics, & AI.

        Our expertise spans Data Engineering, Data Governance, Strategic Analytics, Data Science, and AI. Through our Gen AI capabilities, we also empower physicians, insurers, hospital administrators, and pharmacists to gain precise, contextual insights. These insights encompass real-time demand prediction, staffing optimization, inventory scheduling, automated diagnostic report analysis, preventive healthcare, and more.

        I believe our deep expertise and proven track record can serve as a valuable extension toUCLA Data and AI team.

        Would you like to see a demo of our AI capabilities for hospitals? Or can you connect with someone in your team who would be interested in exploring further?

        Look forward to your response.

        Cheers,
        {your_name}
        Sr. Director of Marketing
        www.factspan.com
        e: {your_contact_info}
        """
    st.header("Email Draft template",divider=True)
    st.write(email_body)
    st.divider()
    return {"output":email_body}

def send_email(state):
    email_body = state["output"]
    llm = ChatOpenAI(
    model = "gpt-4o-mini",temperature=0,streaming=True)
    prompt = PromptTemplate.from_template(

    f"""

    step 1: Your are best Email composer and paraphrase tool. Compose the email and do the following:

    step 2: Get email template include Subject and Body using only ascii charector,

    step 3: provide the output:
    Subject: Provide subject as per {email_body}
    Body: Provide paraphrase body as per {email_body}

    """
    )
    chain = prompt | llm
    response = chain.invoke({"input": email_body})
    email_string = str(response.content)
    from_match = sender_email
    to_match = state["input"]["Email"]
    subject_match = re.search(r'Subject:\s*(.*)', email_string)
    subject = subject_match.group(1) if subject_match else None
    body_match = re.search(r'Body:\n(.*)', email_string, re.DOTALL)
    body = body_match.group(1).strip() if body_match else None
    user_input = st.radio("**Do you like the response?**", ["yes", "no"], index=None, key=f"first_email_{random.randint(1, 1000)}")
    time.sleep(5)
    
    email_data = {}
    if user_input == "yes":
    # Store extracted data in a dictionary
    # if from_match:
        email_data['From'] = from_match
    # if to_match:
        email_data['To'] = to_match
        if subject_match:
            email_data['Subject'] = subject
        if body_match:
            email_data['Body'] = body

        # Print the dictionary
        #print("email_data",email_data)
        #print(email_string)
        receiver_email = state["input"]["Email"]

        # Create the email
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = email_data['Subject']#"Test Email"
        # Email body
        body = email_data['Body']#"This is a test email sent from Python!"
        msg.attach(MIMEText(body, 'plain'))
        # Connect to the server and send the email
        try:
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(sender_email, password)
            text = msg.as_string()
            server.sendmail(sender_email, receiver_email, text)
            st.success('Email sent successfull for the first time!', icon="✅")
            file_name = "./sample_data.csv"
            with open(file_name, 'a', newline='') as file:
                    writer = csv.writer(file)
                    current_time = datetime.now()
                    row_contents = [sender_email,receiver_email,current_time,subject,body]
                    writer.writerow(row_contents)
            file.close()
            email_data = "Email sent successfully!"
        except Exception as e:
            print(f"Error: {e}")
            st.error('Failed to send email!', icon="�")
        finally:
            server.quit()
    return {"email": email_data}


# Function to check for new emails at regular intervals
def sequence_manager(state):
    specific_email_id = state["input"]["Email"]
    last_email_id = None
    latest_email_id = None  # Initialize latest_email_id here

    
    def send_email_again(subject, body, to_email):
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = to_email
        msg['Subject'] = subject
        in_reply_to=None
        msg.attach(MIMEText(body, 'plain'))

        try:
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(sender_email, password)
            text = msg.as_string()
            server.sendmail(sender_email, to_email, text)
            st.success("Response email sent successfully!", icon="✅")
            print(f'Email sent to {to_email} with subject: "{subject}"')
            file_name = "./sample_data.csv"
            with open(file_name, 'a', newline='') as file:
                        writer = csv.writer(file)
                        email_date = msg['Date']
                        print(email_date)
                        current_time = datetime.now()
                        row_contents = [sender_email,to_email,current_time,subject,body]
                        writer.writerow(row_contents)
            file.close()
            print('Email sent successfully')
        except Exception as e:
            print(f'Failed to send email: {e}')
            st.error("Failed to send response email!", icon="�")
        finally:
            server.quit()

    # Function to generate a response using OpenAI's GPT-4
    def generate_response(prompt):
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        response = llm.invoke(prompt)
        return response.content

    # Function to extract plain text from email content
    def extract_plain_text(message):
        if message.is_multipart():
            for part in message.walk():
                if part.get_content_type() == "text/plain":
                    return part.get_payload(decode=True).decode()
        else:
            return message.get_payload(decode=True).decode()
        return ""

    def check_inbox(imap_server, imap_port,sender_email, password,specific_email):
        # Connect to the IMAP server
        mail = imaplib.IMAP4_SSL(imap_server, imap_port)
        mail.login(sender_email, password)
        mail.select('inbox')

        # Search for unseen emails
        status, data = mail.search(None, f'(UNSEEN FROM "{specific_email}")')
        email_ids = data[0].split()
        mail.logout()

        return email_ids

    try:
        new_emails = check_inbox(imap_server, imap_port,sender_email, password,specific_email_id)
        if new_emails:
            print("Current Response!!!")
            mail = imaplib.IMAP4_SSL(imap_server, imap_port)
            mail.login(sender_email, password)
            mail.select('inbox')
            #search_criteria = f'(OR (FROM "{specific_email_id}") (TO "{specific_email_id}"))'
            search_criteria = f'(FROM "{specific_email_id}")'
            result, data = mail.search(None, search_criteria)
            mail_ids = data[0].split()
            latest_email_id = mail_ids[-1]

            if latest_email_id != last_email_id:
                last_email_id = latest_email_id

                try:
                    result, data = mail.fetch(latest_email_id, '(RFC822)')
                    raw_email = data[0][1]
                    msg = email.message_from_bytes(raw_email)
                    email_date = msg['Date']
                    subject = msg['subject']
                    from_email = msg['from']
                    print("from email", from_email)
                    body = extract_plain_text(msg)
                    your_name = "Atul Verma"
                    your_contact_info = "atul.v@factspan.com"

                    if "Re:" in subject:
                        prompt = f"""
                        You are a Senior Marketing Analyst. Reply to the following email:
                        From: {from_email}
                        Subject: {subject}
                        Body: {body}

                        Your reply should acknowledge the sender, address the content of their email,
                        and get back to them with a detailed analysis and aim to convert them into a potential customer..

                        Output only the body of the email in your response. Include the following information at the end:
                        Cheers,
                        {your_name}
                        Sr. Director of Marketing
                        www.factspan.com
                        e: {your_contact_info}
                        """
                        response = generate_response(prompt)
                        # Email body
                        #response = response+"\n"+tmp
                        response = response + "\n\n" + "----- Original Message -----\n" + body
                        print(f'From: {from_email}')
                        print(f'Subject: {subject}')
                        print(f'Body: {response}')
                        st.header("Response Draft for Reply Mail",divider=True)
                        st.write(response)
                        st.divider()
                        user_input1 = st.radio("**Do you like the response?**" ["yes", "no"],index=None,key=f"response_email_{random.randint(1, 1000)}")
                        time.sleep(5)
                        # # Check the input and respond accordingly
                        if user_input1 == "yes":
                            st.write("Glad you liked the response!")
                            send_email_again(f"Re: {subject}", response, from_email)
                            input_context = response
                            prompt = f"""
                            You are a semantic analysis ai tool.
                            Analysis the following Input Context that discussing meeting, action item information.
                            Input Context: {input_context}
                            provide final answer yes or no in lower case.
                            """
                            responseYN = generate_response(prompt)
                            print("Meeting discussion in response: ",response)
                            # send_email_again(f"Re: {subject}", response, from_email)
                        
                            if responseYN == "yes":
                                alert_email = "atul.verma@factspan.com"
                                company_name = state["input"]["Company Name"]
                                contact_name = state["input"]["Contact Name"]
                                job_title = state["input"]["Job Title"]
                                #   email = state["input"]["Email"]
                                phone_number = state["input"]["Phone Number"]
                                industry = state["input"]["Industry"]
                                company_size = state["input"]["Company Size"]
                                location = state["input"]["Location"]

                                response = f"""
                                    Dear,

                                    We wanted to inform you about the following alert from {company_name}:

                                    Company Name: {company_name}
                                    Contact Name: {contact_name}
                                    Job Title: {job_title}
                                    Email: {from_email}
                                    Phone Number: {phone_number}
                                    Industry: {industry}
                                    Company Size: {company_size}
                                    Location: {location}

                                    Please take the necessary actions as soon as possible for following lead for positive reply.

                                    Best regards,
                                    Atul Verma
                                """
                                send_email_again(f"Alert: {subject}", response, alert_email)
                        
                        elif user_input1 == "no":
                            print("You don't like response!")
                            st.write("You don't like the response!")
                        else:
                            print("Please type 'yes' or 'no'.")
                            st.write("Please select an option.")
                
                except imaplib.IMAP4.error as e:
                    print(f"IMAP error: {e}")
                except Exception as e:
                    print(f"General error: {e}")
                finally:
                    if 'mail' in locals() and mail.state != 'LOGOUT':
                        mail.logout()
        else:
            print("Reply Response!!!")
            specific_email_id = state["input"]["Email"]
            search_email = specific_email_id  # Replace with the email you want to search for
            file_name = "./sample_data.csv"
            last_match = None
            with open(file_name, 'r', newline='') as file:
                reader = csv.reader(file)
                # next(reader)  # Uncomment if there's a header row
                for row in reader:
                    if row[1] == search_email:  # Assuming receiver_email is the second column
                        last_match = row
            file.close()
            if last_match:
                receiver_email = last_match[1]
                time_str = last_match[2]  # Assuming the timestamp is the third column
                time_format = "%Y-%m-%d %H:%M:%S.%f"  # Adjust the format to match your timestamp format
                email_time = datetime.strptime(time_str, time_format)
                current_time = datetime.now()
                time_difference = current_time - email_time
                days_difference = time_difference.days
                seconds_in_two_days = 2 * 24 * 60 * 60  # Number of seconds in 2 days
                #if time_difference.total_seconds() % seconds_in_two_days == 0:
                if time_difference.total_seconds() > seconds_in_two_days:
                  #print(f"Receiver Email: {receiver_email}")
                  #print(f"Time: {time_str}")
                  #print("The time is more than 2 days old.")
                  if latest_email_id != last_email_id:
                      last_email_id = latest_email_id
                      try:
                          result, data = mail.fetch(latest_email_id, '(RFC822)')
                          raw_email = data[0][1]
                          msg = email.message_from_bytes(raw_email)
                          email_date = msg['Date']
                          subject = msg['subject']
                          from_email = msg['from']
                          body = extract_plain_text(msg)
                          your_name = "Atul Verma"
                          your_contact_info = "Atul.v@factspan.com"
                          name = state["input"]["Contact Name"]
                          #content = f"Provide all the information for {name} in {company} company?"
                          content = f"any public events, blog post, news articles , linkedln post for {name}"
                          messages = [
                              {"role": "system", "content": "You are an AI assistant."},
                              {"role": "user", "content": content}
                          ]
                          from openai import OpenAI
                          YOUR_API_KEY = "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"
                          client = OpenAI(api_key=YOUR_API_KEY, base_url="https://api.perplexity.ai")
                          response = client.chat.completions.create(
                              model="llama-3.1-sonar-small-128k-online",
                              #model="llama-3.1-sonar-small-128k-chat",
                              messages=messages,
                          )
                          input_context = response.choices[0].message.content
                          if "Re:" in subject:
                            prompt = f"""
                            You are a Senior Marketing Analyst. Reply to the following email:
                            From: {from_email}
                            Subject: {subject}
                            Body: {body}
                            Input Context: {input_context}
                            Your reply should acknowledge the sender, address the content of their email,
                            and get back to them with a detailed analysis.
                            Consider the input context along with the body to craft your response.
                            Output only the body of the email in your response. Include the following information at the end:
                            Cheers,
                            {your_name}
                            Sr. Director of Marketing
                            www.factspan.com
                            e: {your_contact_info}
                            """
                            response = generate_response(prompt)
                            # Email body
                            #response = response+"\n"+tmp
                            response = response + "\n\n" + "----- Original Message -----\n" + body
                            print(f'From: {from_email}')
                            print(f'Subject: {subject}')
                            print(f'Body: {response}')
                            send_email_again(f"Re: {subject}", response, from_email)
                            st.header("Response Email",divider=True)
                            st.write(response)
                            st.success("Reply mail sent successfully!", icon="✅")
                      except imaplib.IMAP4.error as e:
                          print(f"IMAP error: {e}")
                      except Exception as e:
                          print(f"General error: {e}")
                      finally:
                          if 'mail' in locals() and mail.state != 'LOGOUT':
                              mail.logout()
    except Exception as e:
            print(f'Failed to check for new emails: {e}')
            st.error("Failed to check for new emails", icon="🚨")


def follow_up(state):
    print("Follow Up Response!!!")

    # Function to check if the time difference is more than 5 minutes
    def is_time_difference_more_than_5_minutes(email_date):
        email_time = parsedate_to_datetime(email_date)
        current_time = datetime.now(email_time.tzinfo)
        return (current_time - email_time) > timedelta(minutes=5)

    # Function to check if there is a reply from the specific user email
    def has_reply_from_specific_user(mail, specific_user_email):
        mail.select('inbox')
        result, data = mail.search(None, f'FROM "{specific_user_email}"')
        return len(data[0].split()) > 0

    specific_email_id = state["input"]["Email"]
    mail = imaplib.IMAP4_SSL('imap.gmail.com')
    mail.login(sender_email, password)
    specific_user_email = specific_email_id
    # Check for replies from the specific user email
    if not has_reply_from_specific_user(mail, specific_user_email):
    # Select the 'Sent' mailbox
        mail.select('"[Gmail]/Sent Mail"')
        # Search for emails sent to a specific user
        result, data = mail.search(None, f'TO "{specific_user_email}"')
        # Fetch the email content
        email_ids = data[0].split()
        if email_ids:
            latest_email_id = email_ids[-1] # Get the latest email ID
            result, msg_data = mail.fetch(latest_email_id, '(RFC822)')
            raw_email = msg_data[0][1]
            msg = email.message_from_bytes(raw_email)
            # Check the email date
            email_date = msg['Date']
            email_time = parsedate_to_datetime(email_date)
            current_time = datetime.now(email_time.tzinfo)
            time_difference = current_time - email_time
            search_email = specific_user_email  # Replace with the email you want to search for
            last_match = None
            file_name = "./sample_data.csv"
            with open(file_name, 'r', newline='') as file:
                reader = csv.reader(file)
                # next(reader)  # Uncomment if there's a header row
                for row in reader:
                    if row[1] == search_email:  # Assuming receiver_email is the second column
                        last_match = row
            file.close()
            if last_match:
                receiver_email = last_match[1]
                time_str = last_match[2]  # Assuming the timestamp is the third column
                time_format = "%Y-%m-%d %H:%M:%S.%f"  # Adjust the format to match your timestamp format
                email_time = datetime.strptime(time_str, time_format)
                current_time = datetime.now()
                time_difference = current_time - email_time
                days_difference = time_difference.days
                seconds_in_two_days = 2 * 24 * 60 * 60  # Number of seconds in 2 days
                #if time_difference.total_seconds() % seconds_in_two_days == 0:
                if time_difference.total_seconds() > seconds_in_two_days:
                    #print(f"Receiver Email: {receiver_email}")
                    #print(f"Time: {time_str}")
                    #print("The time is more than 2 days old.")
                    #print(f'Subject: {msg["Subject"]}, From: {msg["From"]}, To: {msg["To"]}')
                    #print(f'Email Date: {email_time}, Current Time: {current_time}, Time Difference: {time_difference}')
                    # Print the email body
                    if msg.is_multipart():
                        for part in msg.walk():
                            if part.get_content_type() == 'text/plain':
                                #print(part.get_payload(decode=True).decode())
                                email_body = part.get_payload(decode=True).decode()

                    else:
                        #print(msg.get_payload(decode=True).decode())
                        email_body = msg.get_payload(decode=True).decode()

                    name = state["input"]["Contact Name"]
                    #content = f"Provide all the information for {name} in {company} company?"
                    content = f"any public events, blog post, news articles , linkedln post for {name}"
                    messages = [
                        {"role": "system", "content": "You are an AI assistant."},
                        {"role": "user", "content": content}
                    ]
                    from openai import OpenAI
                    YOUR_API_KEY = "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"
                    client = OpenAI(api_key=YOUR_API_KEY, base_url="https://api.perplexity.ai")
                    response = client.chat.completions.create(
                        model="llama-3.1-sonar-small-128k-online",
                        #model="llama-3.1-sonar-small-128k-chat",
                        messages=messages,
                    )
                    input_context = response.choices[0].message.content

                    llm = ChatOpenAI(
                    model = "gpt-4o-mini",temperature=0,streaming=True)
                    prompt = PromptTemplate.from_template(

                    f"""

                    step 1: Your are best Email composer and paraphrase tool.
                            Consider the input context along with the body to craft your response.
                            Compose the email and do the following:
                            Input Context: {input_context}
                    step 2: Get email template include Subject and Body using only ascii charector and
                    step 3: provide the output:
                    Subject: Provide subject as per {email_body}
                    Body: Provide paraphrase body as per {email_body}

                    """
                    )
                    chain = prompt | llm
                    response = chain.invoke({"input": email_body})
                    email_string = str(response.content)
                    #print(f'Subject: {msg["Subject"]}, From: {msg["From"]}, To: {msg["To"]}')
                    
                    from_match = sender_email
                    to_match = state["input"]["Email"]

                    # Extract Subject
                    subject_match = re.search(r'Subject:\s*(.*)', email_string)
                    subject = subject_match.group(1) if subject_match else None

                    # Extract Body
                    body_match = re.search(r'Body:\s*([\s\S]*)', email_string)
                    body = body_match.group(1) if body_match else None

                    print("Subject:", subject)
                    print("Body:", body)
                    # Store extracted data in a dictionary
                    email_data = {}
                    # if from_match:
                    email_data['From'] = from_match
                    # if to_match:
                    email_data['To'] = to_match
                    if subject_match:
                        email_data['Subject'] = subject
                    if body_match:
                        email_data['Body'] = body

                    # Print the dictionary
                    #print("email_data",email_data)
                    #print(email_string)
                    receiver_email = state["input"]["Email"]
                    # Create the email
                    msg = MIMEMultipart()
                    msg['From'] = sender_email
                    msg['To'] = receiver_email
                    msg['Subject'] = email_data['Subject']
                    # Email body
                    #   body = email_data['Body']
                    #   msg.attach(MIMEText(body, 'plain'))
                    # Connect to the server and send the email
                    try:
                        server = smtplib.SMTP('smtp.gmail.com', 587)
                        server.starttls()
                        server.login(sender_email, password)
                        text = msg.as_string()
                        server.sendmail(sender_email, receiver_email, text)
                        st.success("Email sent successfully!", icon="✅")
                        file_name = "./sample_data.csv"

                        with open(file_name, 'a', newline='') as file:
                                    writer = csv.writer(file)
                                    current_time = datetime.now()
                                    row_contents = [sender_email,receiver_email,current_time,subject,body]
                                    writer.writerow(row_contents)
                        file.close()
                        print("Email sent successfully!")
                        #time.sleep(60)
                        email_data = "Email sent successfully!"
                    except Exception as e:
                        print(f"Error: {e}")
                    finally:
                        server.quit()
        else:
            print("No emails found from the specified user.")
    mail.logout()

#You can precise the format here which could be helpfull for multimodal graphs

class AgentState(TypedDict):
    input: dict
    output: str
    decision: str
    email: dict
    count : int

workflow = StateGraph(AgentState)
workflow.add_node("check_email_status", check_email_status)
workflow.add_node("research_agent", analyze_question)
workflow.add_node("crm_integration_agent", crm_integration_agent)
workflow.add_node("public_agent", public_agent)
workflow.add_node("customer_intelligence_report_agent", customer_intelligence_report)
workflow.add_node("compose_email", compose_email)
workflow.add_node("send_email", send_email)
workflow.add_node("sequence_manager", sequence_manager)
workflow.add_node("follow_up", follow_up)

workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "check_email_status",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # Finally we pass in a mapping.
    # The keys are strings, and the values are other nodes.
    # END is a special node marking that the graph should finish.
    # What will happen is we will call `should_continue`, and then the output of that
    # will be matched against the keys in this mapping.
    # Based on which one it matches, that node will then be called.
    {
        # If `tools`, then we call the tool node.
        "continue": "research_agent",
        # Otherwise we finish.
        "conversation handler": "sequence_manager"
    }
)

workflow.add_conditional_edges(
    "research_agent",
    lambda x: x["decision"],
    {
        "crm_integration_agent": "crm_integration_agent",
        "public_agent": "public_agent"
    }
)

workflow.set_entry_point("check_email_status")
workflow.add_edge("crm_integration_agent", "customer_intelligence_report_agent")
workflow.add_edge("public_agent", "customer_intelligence_report_agent")
workflow.add_edge("customer_intelligence_report_agent", "compose_email")
workflow.add_edge("compose_email", "send_email")
workflow.add_edge("send_email", "sequence_manager")
workflow.add_edge("sequence_manager", "follow_up")
workflow.set_finish_point("follow_up")
app = workflow.compile()


c=0
def my_function():
    print("##################")
    c=0
    if len(lead_list)>0:
        for lead in lead_list:
            c+=1
            print("Input Lead: ",c)
            input = {"input":lead}
            for output in app.stream(input):
                for key, value in output.items():
                    print(f"Output from graph key node '{key}': ")
                    print("---")
                    print(f"Output from graph value node: ")
                    print("---")
                    print(value)
            print("\n---\n")
            print("##################")
    else:
        print("No leads found!")
        st.write("No leads found! , Please add Leads in the input box!")

while True:
    print("####################################")
    c=c+1
    print("Graph Iteration: ",c)
    my_function()
    with st.spinner('Wait for some time to fetch again...'):
        time.sleep(10)
    print("####################################")
