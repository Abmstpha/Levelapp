import { Metadata } from "next";
import Link from "next/link";
import UserAuthForm from "./user-auth-form";
import { buttonVariants } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export const metadata: Metadata = {
  title: "Authentication",
  description: "Authentication forms built using the components.",
};

export default function SignInViewPage() {
  return (
    <div className="relative h-screen flex-col items-center justify-center md:grid lg:max-w-none lg:grid-cols-2 lg:px-0">
      <Link
        href="/examples/authentication"
        className={cn(
          buttonVariants({ variant: "ghost" }),
          "absolute right-4 top-4 hidden md:right-8 md:top-8"
        )}
      >
        Login
      </Link>
      <div className="relative hidden h-full flex-col bg-muted p-10 text-white lg:flex dark:border-r">
        <div className="absolute inset-0 bg-zinc-900 bg-[url('https://norma.dev/assets/bg-fixed.svg')] bg-cover bg-no-repeat" />
        <div className="relative z-20 flex items-center text-lg font-medium">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="164"
            height="36"
            viewBox="0 0 164 36"
            fill="none"
          >
            <g clipPath="url(#clip0_575_1222)">
              <path
                d="M63.3252 24.794V10.7974H64.6012L74.7077 22.955H74.0385V10.7974H75.5661V24.794H74.2901L64.2051 12.6364H64.8743V24.794H63.3252Z"
                fill="white"
              />
              <path
                d="M91.3541 24.9133C90.2518 24.9133 89.2275 24.7393 88.2773 24.3934C87.3428 24.0338 86.5272 23.5336 85.8287 22.8945C85.1459 22.2418 84.6074 21.4874 84.2172 20.6353C83.8406 19.7696 83.6514 18.8218 83.6514 17.7958C83.6514 16.7697 83.8406 15.8297 84.2172 14.9757C84.6074 14.11 85.1439 13.3556 85.8287 12.7165C86.5253 12.0638 87.3428 11.5635 88.2773 11.2176C89.2119 10.858 90.2381 10.6782 91.3541 10.6782C92.4701 10.6782 93.4749 10.858 94.4095 11.2176C95.3441 11.5635 96.1537 12.058 96.8366 12.697C97.5331 13.3361 98.0716 14.0904 98.4482 14.9562C98.8384 15.8219 99.0335 16.7697 99.0335 17.7958C99.0335 18.8218 98.8384 19.7676 98.4482 20.6353C98.0716 21.5011 97.5351 22.2554 96.8366 22.8945C96.1537 23.5336 95.3441 24.0338 94.4095 24.3934C93.4749 24.7393 92.4565 24.9133 91.3541 24.9133ZM91.3541 23.5941C92.2321 23.5941 93.0418 23.4534 93.7813 23.174C94.5344 22.8808 95.1841 22.4743 95.7265 21.9545C96.2845 21.421 96.7176 20.8073 97.0239 20.1155C97.3302 19.408 97.4844 18.6361 97.4844 17.7958C97.4844 16.9554 97.3302 16.1893 97.0239 15.4956C96.7176 14.7901 96.2845 14.1764 95.7265 13.6566C95.1821 13.123 94.5344 12.7165 93.7813 12.4371C93.0418 12.1439 92.2321 11.9974 91.3541 11.9974C90.4762 11.9974 89.6587 12.1439 88.9056 12.4371C88.1525 12.7165 87.4969 13.123 86.9389 13.6566C86.3946 14.1764 85.9634 14.7901 85.6414 15.4956C85.3351 16.1893 85.181 16.9554 85.181 17.7958C85.181 18.6361 85.3351 19.3885 85.6414 20.096C85.9614 20.8015 86.3946 21.4229 86.9389 21.9545C87.4969 22.4743 88.1525 22.8808 88.9056 23.174C89.6587 23.4534 90.4742 23.5941 91.3541 23.5941Z"
                fill="white"
              />
              <path
                d="M107.109 24.794V10.7974H112.592C113.833 10.7974 114.9 10.9908 115.794 11.3778C116.687 11.7511 117.37 12.2983 117.844 13.0174C118.332 13.7229 118.578 14.5828 118.578 15.5971C118.578 16.6114 118.334 17.4361 117.844 18.1572C117.37 18.8627 116.687 19.4099 115.794 19.7969C114.9 20.1701 113.833 20.3558 112.592 20.3558H107.968L108.659 19.6757V24.794H107.109ZM117.112 24.794L113.347 19.7148H115.021L118.808 24.794H117.112ZM108.659 19.7949L107.968 19.0953H112.551C114.016 19.0953 115.124 18.7885 115.877 18.1748C116.644 17.5612 117.029 16.7013 117.029 15.5951C117.029 14.489 116.644 13.6096 115.877 12.9959C115.124 12.3823 114.014 12.0755 112.551 12.0755H107.968L108.659 11.3758V19.793V19.7949Z"
                fill="white"
              />
              <path
                d="M126.834 24.794V10.7974H128.11L134.806 21.714H134.135L140.768 10.7974H142.044V24.794H140.558V13.0174H140.915L134.804 23.0156H134.07L127.919 13.0174H128.317V24.794H126.83H126.834Z"
                fill="white"
              />
              <path
                d="M149.08 24.794L155.776 10.7974H157.304L164 24.794H162.367L156.215 11.6573H156.843L150.692 24.794H149.08ZM151.716 21.0535L152.176 19.8535H160.693L161.153 21.0535H151.716Z"
                fill="white"
              />
              <path
                d="M32.7308 9.10108C32.7308 13.934 32.7269 18.7651 32.7562 23.598C32.7621 24.4442 32.7289 25.4898 31.8782 25.9314C30.1535 26.8265 28.579 23.8853 27.902 22.7088C26.308 19.9396 25.6817 16.7912 26.0368 13.6135C26.3041 11.241 27.0396 8.9506 27.9995 6.77743C28.7156 5.15537 29.5916 3.40432 31.0178 2.26497C32.6782 0.936053 34.9433 0.758212 36.6154 2.17898C38.4065 3.70137 38.4591 6.45302 38.4377 8.63987C38.3694 15.4858 38.5469 22.3434 38.3811 29.1892C38.3616 29.9925 38.3284 30.8074 38.1021 31.5793C37.2319 34.5342 33.5503 35.527 30.9866 34.3036C29.4687 33.5786 28.2746 32.3083 27.2269 30.9852C23.8438 26.7112 21.8732 21.6594 19.2588 16.9437"
                stroke="white"
                stroke-width="2"
                stroke-miterlimit="10"
                strokeLinecap="round"
              />
              <path
                d="M6.90709 26.9009C6.90709 22.068 6.91099 17.237 6.88172 12.404C6.87587 11.5578 6.90904 10.5122 7.7597 10.0706C9.48443 9.17551 11.0589 12.1167 11.736 13.2932C13.33 16.0624 13.9563 19.2108 13.6012 22.3885C13.3339 24.761 12.5983 27.0514 11.6384 29.2246C10.9224 30.8466 10.0463 32.5977 8.62011 33.737C6.95976 35.066 4.69459 35.2438 3.02253 33.823C1.23146 32.3006 1.17878 29.547 1.20024 27.3602C1.26853 20.5143 1.09098 13.6567 1.25682 6.81082C1.27633 6.0076 1.3095 5.19266 1.53582 4.42072C2.40795 1.46583 6.08764 0.475005 8.65328 1.69839C10.1712 2.42343 11.3653 3.69372 12.413 5.01678C15.7961 9.29081 17.7667 14.3427 20.3811 19.0584"
                stroke="white"
                stroke-width="2"
                stroke-miterlimit="10"
                strokeLinecap="round"
              />
            </g>
            <defs>
              <clipPath id="clip0_575_1222">
                <rect width="164" height="36" fill="white" />
              </clipPath>
            </defs>
          </svg>
        </div>
        <div className="relative z-20 mt-auto">
          <blockquote className="space-y-2">
            <p className="text-lg">
              Welcome to Norma's Evaluation Hub — the ultimate platform to
              benchmark, analyze, and elevate the performance of cutting-edge
              LLM agents built for your success.
            </p>
          </blockquote>
        </div>
      </div>
      <div className="flex h-full items-center p-4 lg:p-8">
        <div className="mx-auto flex w-full flex-col justify-center space-y-6 sm:w-[350px]">
          <div className="flex flex-col space-y-2 text-center">
            <h1 className="text-2xl font-semibold tracking-tight">
              Sign into your account
            </h1>
            <p className="text-sm text-muted-foreground">
              Enter your email below to log in your account
            </p>
          </div>
          <UserAuthForm />
          <p className="px-8 text-center text-sm text-muted-foreground">
            <span>Terms of Service</span> and <span>Privacy Policy</span>.
          </p>
        </div>
      </div>
    </div>
  );
}
